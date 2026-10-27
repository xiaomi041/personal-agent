# -*- coding: utf-8 -*-
"""
核心逻辑的自动化测试（不联网、不调用任何模型接口、不碰你的真实数据）。

运行（在项目根目录）：
    python -m unittest discover -s tests -v
也兼容 pytest：pytest tests -v
"""
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")  # main.py 缺 key 会直接退出，测试用假值
os.environ.setdefault("QWEN_API_KEY", "test-key")

import main  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

_REAL_EMBED = main._embed_texts  # 在任何 mock 之前保存原函数


class IsolatedCase(unittest.TestCase):
    """每个用例使用独立临时目录，所有读写都落在里面。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.desktop = self.tmp / "Desktop"
        self.desktop.mkdir()
        kb = self.tmp / "kb"
        patches = [
            mock.patch.dict(os.environ, {"HOME": str(self.tmp), "USERPROFILE": str(self.tmp)}),
            mock.patch.object(main, "DB_FILE", self.tmp / "agent.db"),
            mock.patch.object(main, "MEMO_FILE", self.desktop / "memos.md"),
            mock.patch.object(main, "SESSIONS_FILE", self.tmp / "sessions.json"),
            mock.patch.object(main, "TODO_FILE", self.tmp / "todos.json"),
            mock.patch.object(main, "PLANS_FILE", self.tmp / "plans.json"),
            mock.patch.object(main, "KB_DIR", kb),
            mock.patch.object(main, "KB_CHUNKS", kb / "chunks.json"),
            mock.patch.object(main, "KB_VECS", kb / "vectors.npy"),
            mock.patch.object(main, "KB_DOCS", kb / "docs.json"),
            mock.patch.object(main, "_notify_desktop", lambda *a, **k: None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

    def alerts(self, like):
        conn = main._db()
        try:
            return conn.execute("SELECT COUNT(*) FROM alerts WHERE content LIKE ?", (f"%{like}%",)).fetchone()[0]
        finally:
            conn.close()


class ParseAtTests(unittest.TestCase):
    def test_once_with_seconds(self):
        self.assertEqual(main._parse_at("2026-10-27 08:00:30", "once")[0].second, 30)

    def test_once_minute_precision_still_supported(self):
        self.assertEqual(main._parse_at("2026-10-27 08:00", "once")[0].minute, 0)

    def test_once_bad_format_raises(self):
        with self.assertRaises(ValueError):
            main._parse_at("明天8点", "once")

    def test_weekly_variants(self):
        self.assertEqual(main._parse_at("周三 08:00", "weekly")[1], 2)
        self.assertEqual(main._parse_at("星期天 09:30", "weekly")[1], 6)
        self.assertEqual(main._parse_at("wed 08:00", "weekly")[1], 2)  # 旧实现 lstrip 会把 wed 剥成 d

    def test_daily(self):
        self.assertEqual(main._parse_at("08:30", "daily")[2], "08:30")


class ReminderTests(IsolatedCase):
    def test_daily_fires_once_per_minute_despite_repeated_polls(self):
        main.create_reminder.invoke({"content": "喝水daily", "at": datetime.now().strftime("%H:%M"), "repeat": "daily"})
        for _ in range(4):  # 模拟一分钟内多次轮询
            main._check_due()
        self.assertEqual(self.alerts("喝水daily"), 1)

    def test_once_in_the_past_fires_and_is_removed(self):
        at = (datetime.now() - timedelta(seconds=3)).strftime("%Y-%m-%d %H:%M:%S")
        main.create_reminder.invoke({"content": "过期once", "at": at, "repeat": "once"})
        main._check_due()
        main._check_due()
        self.assertEqual(self.alerts("过期once"), 1)
        self.assertEqual(main.list_reminders.invoke({}), "当前没有未执行的定时提醒")

    def test_once_in_the_future_does_not_fire(self):
        at = (datetime.now() + timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")
        main.create_reminder.invoke({"content": "未来once", "at": at, "repeat": "once"})
        main._check_due()
        self.assertEqual(self.alerts("未来once"), 0)

    def test_old_database_gets_last_fired_column(self):
        conn = sqlite3.connect(main.DB_FILE)
        conn.execute("CREATE TABLE reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL, "
                     "at TEXT NOT NULL, repeat TEXT NOT NULL DEFAULT 'once', created_at TEXT NOT NULL)")
        conn.commit()
        conn.close()
        conn = main._db()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(reminders)")]
        conn.close()
        self.assertIn("last_fired", cols)


class AlertHttpServerTests(IsolatedCase):
    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), main._AlertHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        conn = main._db()
        conn.execute("INSERT INTO alerts (content, fired_at) VALUES ('测试提醒', '2026-10-27 08:00')")
        conn.commit()
        conn.close()

    def test_get_alerts_has_no_cors_header(self):
        r = urllib.request.urlopen(f"{self.base}/alerts")
        self.assertEqual(len(json.loads(r.read())), 1)
        self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))

    def test_ack_requires_post(self):
        aid = json.loads(urllib.request.urlopen(f"{self.base}/alerts").read())[0]["id"]
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(f"{self.base}/ack?id={aid}")
        self.assertEqual(cm.exception.code, 404)
        self.assertEqual(len(main.get_unread_alerts()), 1)  # GET 没有改变状态
        urllib.request.urlopen(urllib.request.Request(f"{self.base}/ack?id={aid}", method="POST"))
        self.assertEqual(len(main.get_unread_alerts()), 0)


class NotifyTests(unittest.TestCase):
    def test_noop_when_not_windows(self):
        with mock.patch.object(main, "_is_windows", return_value=False), mock.patch("subprocess.Popen") as popen:
            main._notify_desktop("t", "x")
        popen.assert_not_called()

    def test_can_be_disabled_by_env(self):
        with mock.patch.object(main, "_is_windows", return_value=True), \
                mock.patch.dict(os.environ, {"AGENT_DESKTOP_NOTIFY": "0"}), mock.patch("subprocess.Popen") as popen:
            main._notify_desktop("t", "x")
        popen.assert_not_called()

    def test_single_quotes_are_escaped_for_powershell(self):
        with mock.patch.object(main, "_is_windows", return_value=True), mock.patch("subprocess.Popen") as popen:
            main._notify_desktop("标题", "it's $(calc) 换行\n第二行")
        script = popen.call_args[0][0][-1]
        self.assertIn("'it''s $(calc) 换行 第二行'", script)  # 单引号写两次，$() 在单引号串里不会被执行
        self.assertEqual(popen.call_args[0][0][0], "powershell")

    def test_popen_failure_is_swallowed(self):
        with mock.patch.object(main, "_is_windows", return_value=True), \
                mock.patch("subprocess.Popen", side_effect=OSError("no powershell")):
            main._notify_desktop("t", "x")  # 不应抛异常


class ToolTests(IsolatedCase):
    def test_write_file_is_confined_to_desktop(self):
        main.write_file.invoke({"filename": "../../escape.md", "content": "a"})
        main.write_file.invoke({"filename": "C:\\Windows\\evil.txt", "content": "a"})
        self.assertTrue((self.desktop / "escape.md").exists())
        self.assertTrue((self.desktop / "evil.txt").exists())
        self.assertFalse((self.tmp.parent / "escape.md").exists())

    def test_write_file_does_not_overwrite(self):
        main.write_file.invoke({"filename": "dup.md", "content": "v1"})
        main.write_file.invoke({"filename": "dup.md", "content": "v2"})
        self.assertEqual((self.desktop / "dup.md").read_text(encoding="utf-8"), "v1")
        self.assertTrue((self.desktop / "dup(1).md").exists())

    def test_write_file_rejects_unknown_extension(self):
        self.assertIn("不支持", main.write_file.invoke({"filename": "a.exe", "content": "x"}))

    def test_todo_roundtrip(self):
        main.add_todo.invoke({"content": "交房租"})
        main.complete_todo.invoke({"index": 1})
        self.assertTrue(main.todo_items()[0]["done"])

    def test_active_thread_is_per_thread(self):
        seen = {}

        def worker(tid):
            main._ctx.thread_id = tid
            time.sleep(0.1)
            seen[tid] = main._active_thread()

        ts = [threading.Thread(target=worker, args=(t,)) for t in ("A", "B")]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(seen, {"A": "A", "B": "B"})


class GraphLogicTests(unittest.TestCase):
    def test_planner_resets_stale_review_state(self):
        fake = mock.Mock()
        fake.invoke.return_value = AIMessage("计划：①x")
        with mock.patch.object(main, "deepseek_model", return_value=fake):
            out = main.planner_node({"messages": [HumanMessage("hi")], "review": "需修正", "retry_count": 23})
        self.assertEqual((out["review"], out["retry_count"], out["execution_trace"], out["execution_result"]), ("", 0, "", ""))

    def test_router(self):
        R = main._review_router
        self.assertEqual(R({"review": "需修正\n- 问题", "retry_count": 1}), "executor")
        self.assertEqual(R({"review": "需修正", "retry_count": main.MAX_RETRY}), "executor")  # 允许 MAX_RETRY 次返工
        self.assertEqual(R({"review": "需修正", "retry_count": main.MAX_RETRY + 1}), "end")
        self.assertEqual(R({"review": "通过", "retry_count": 1}), "end")
        self.assertEqual(R({"review": "通过，无需修正", "retry_count": 1}), "end")

    def test_needs_fix_variants(self):
        self.assertTrue(main._needs_fix("结论：需修正\n..."))
        self.assertFalse(main._needs_fix("不需修正"))
        self.assertFalse(main._needs_fix(""))

    def test_context_is_limited_to_recent_turns(self):
        msgs = []
        for i in range(main.MAX_CONTEXT_TURNS + 8):
            msgs += [HumanMessage(f"问题{i}"), AIMessage(f"答案{i}")]
        ctx = main._build_context({"messages": msgs})
        self.assertEqual(ctx.count("用户："), main.MAX_CONTEXT_TURNS)
        self.assertIn(f"问题{main.MAX_CONTEXT_TURNS + 7}", ctx)
        self.assertNotIn("问题0\n", ctx)

    def test_internal_node_messages_are_excluded_from_context(self):
        ctx = main._build_context({"messages": [HumanMessage("你好"), AIMessage("[规划] 计划：①"), AIMessage("[评审] 通过"), AIMessage("你好呀")]})
        self.assertNotIn("[规划]", ctx)
        self.assertNotIn("[评审]", ctx)
        self.assertIn("助手：你好呀", ctx)


class KnowledgeBaseTests(IsolatedCase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(main, "_embed_texts", lambda texts: [[float(i % 3), 1.0, 0.0] for i, _ in enumerate(texts)])
        p.start()
        self.addCleanup(p.stop)

    def test_chunking(self):
        self.assertEqual(main._chunk_text("段一\n\n段二"), ["段一", "段二"])
        long = "字" * 800
        chunks = main._chunk_text(long)
        self.assertTrue(all(len(c) <= 350 for c in chunks))
        self.assertGreater(len(chunks), 2)

    def test_add_delete_keeps_chunks_and_vectors_aligned(self):
        main.add_document("a.md", ("段落一内容足够长足够长足够长\n\n段落二内容足够长足够长足够长").encode(), thread_id="T1")
        main.add_document("b.md", ("另一会话的文档内容足够长" * 4).encode(), thread_id="T2")
        chunks, vecs, docs = main._load_kb()
        self.assertEqual(len(chunks), len(vecs))
        self.assertEqual([p for p in os.listdir(main.KB_DIR) if "tmp" in p], [])  # 原子写不留临时文件
        main.register_session("T1", "t1")
        main.delete_session("T1")  # 删除会话应清理它的文档，且不影响其它会话
        chunks, vecs, docs = main._load_kb()
        self.assertEqual({c["thread_id"] for c in chunks}, {"T2"})
        self.assertEqual((len(chunks), len(vecs), len(docs)), (1, 1, 1))

    def test_search_is_scoped_to_thread(self):
        main.add_document("a.md", ("会话一的文档内容足够长" * 4).encode(), thread_id="T1")
        self.assertEqual(main.search_documents("文档", thread_id="T2"), [])
        self.assertGreater(len(main.search_documents("文档", thread_id="T1")), 0)


class EmbedRetryTests(unittest.TestCase):
    def _resp(self):
        ok = mock.Mock()
        ok.json.return_value = {"data": [{"index": 0, "embedding": [1.0]}]}
        ok.raise_for_status.return_value = None
        return ok

    def test_retries_then_succeeds(self):
        import requests

        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.ConnectionError("boom")
            return self._resp()

        with mock.patch.object(main.requests, "post", flaky), mock.patch.object(main._time, "sleep", lambda s: None):
            self.assertEqual(_REAL_EMBED(["a"]), [[1.0]])
        self.assertEqual(calls["n"], 3)

    def test_gives_up_after_three_attempts(self):
        import requests

        calls = {"n": 0}

        def always_fail(*a, **k):
            calls["n"] += 1
            raise requests.Timeout("slow")

        with mock.patch.object(main.requests, "post", always_fail), mock.patch.object(main._time, "sleep", lambda s: None):
            with self.assertRaises(requests.Timeout):
                _REAL_EMBED(["a"])
        self.assertEqual(calls["n"], 3)


class SessionTests(IsolatedCase):
    def test_pinned_sessions_sort_first(self):
        main.SESSIONS_FILE.write_text(json.dumps([
            {"thread_id": "n1", "created_at": "2026-10-02", "title": "n"},
            {"thread_id": "p1", "created_at": "2026-10-01", "title": "p", "pinned": True},
        ]), encoding="utf-8")
        self.assertEqual(main.list_sessions()[0]["thread_id"], "p1")

    def test_purge_empty_sessions_keeps_used_current_and_pinned(self):
        conn = sqlite3.connect(main.DB_FILE)
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT)")
        conn.execute("INSERT INTO checkpoints VALUES ('used', '', '1')")
        conn.commit()
        conn.close()
        main.SESSIONS_FILE.write_text(json.dumps([
            {"thread_id": "used", "title": "聊过的"},
            {"thread_id": "empty1", "title": "新会话"},
            {"thread_id": "current", "title": "新会话"},
            {"thread_id": "pinned", "title": "新会话", "pinned": True},
        ]), encoding="utf-8")
        self.assertEqual(main.purge_empty_sessions(keep="current"), 1)
        self.assertEqual({r["thread_id"] for r in main._load_sessions()}, {"used", "current", "pinned"})


class PruneCheckpointTests(IsolatedCase):
    def test_prune_keeps_latest_per_thread_and_drops_orphan_writes(self):
        conn = sqlite3.connect(main.DB_FILE)
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT)")
        conn.execute("CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT)")
        for tid in ("A", "B"):
            for i in range(1, 7):
                conn.execute("INSERT INTO checkpoints VALUES (?, '', ?)", (tid, f"{i:03d}"))
                conn.execute("INSERT INTO writes VALUES (?, '', ?)", (tid, f"{i:03d}"))
        conn.commit()
        conn.close()
        self.assertEqual(main.prune_checkpoints(keep=2), 8)
        conn = sqlite3.connect(main.DB_FILE)
        self.assertEqual(conn.execute("SELECT checkpoint_id FROM checkpoints WHERE thread_id='A' ORDER BY 1").fetchall(), [("005",), ("006",)])
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0], 4)
        conn.close()


if __name__ == "__main__":
    unittest.main()
