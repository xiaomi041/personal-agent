# -*- coding: utf-8 -*-
"""定时提醒与自动任务：SQLite 持久化、后台调度线程、本地提醒 HTTP 服务。"""
import json
import logging
import sqlite3
import threading
import time as _time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler

from langchain_core.tools import tool

from runtime import call_llm
from settings import *  # noqa: F401,F403
from store import _load_plans, list_todos
from weather import get_weather

log = logging.getLogger("agent.scheduler")

def _db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            at TEXT NOT NULL,
            repeat TEXT NOT NULL DEFAULT 'once',
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            fired_at TEXT NOT NULL,
            shown INTEGER NOT NULL DEFAULT 0
        )"""
    )
    # 兼容旧库：补上动作型提醒需要的列（已存在则跳过）
    for col_sql in (
        "ALTER TABLE reminders ADD COLUMN action TEXT DEFAULT ''",
        "ALTER TABLE reminders ADD COLUMN action_arg TEXT DEFAULT ''",
    ):
        try:
            conn.execute(col_sql)
        except sqlite3.OperationalError:
            pass
    return conn

def _parse_at(at: str, repeat: str):
    """把模型给的时间格式解析成 (datetime, 星期几, HH:MM)。"""
    if repeat == "once":
        return datetime.strptime(at.strip(), "%Y-%m-%d %H:%M"), None, None
    if repeat == "daily":
        hm = at.strip()
        datetime.strptime(hm, "%H:%M")  # 校验格式
        return None, None, hm
    if repeat == "weekly":
        parts = at.strip().split()
        if len(parts) != 2:
            raise ValueError("weekly 格式应为：周X HH:MM")
        wd = parts[0].lower().lstrip("周星期礼拜week")
        if wd in _WEEKDAY_MAP:
            wd = _WEEKDAY_MAP[wd]
        else:
            raise ValueError(f"无法识别星期：{parts[0]}")
        datetime.strptime(parts[1], "%H:%M")
        return None, wd, parts[1]
    raise ValueError(f"不支持的 repeat 类型：{repeat}")

@tool
def create_reminder(content: str, at: str, repeat: str = "once") -> str:
    """创建定时提醒/自动任务，到点后自动写入桌面备忘并推送到界面。
    repeat=once 时 at 为具体时间 "YYYY-MM-DD HH:MM"（如 2026-10-13 08:00）；
    repeat=daily 时 at 为 "HH:MM"（每天这个时刻）；
    repeat=weekly 时 at 为 "周X HH:MM"（每周这一天这个时刻）。
    用户说"每天/每周/明早/后天 X 点提醒我 XX"时，先把相对时间换算成上述格式再调用。"""
    from datetime import timedelta

    _parse_at(at, repeat)
    conn = _db()
    conn.execute(
        "INSERT INTO reminders (content, at, repeat, created_at) VALUES (?,?,?,?)",
        (content, at, repeat, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()
    r = {"once": "一次性", "daily": "每天", "weekly": "每周"}[repeat]
    return f"已创建{r}提醒：{at} → {content}"

@tool
def list_reminders() -> str:
    """列出所有尚未执行的定时提醒。"""
    conn = _db()
    rows = conn.execute("SELECT id, content, at, repeat FROM reminders ORDER BY at").fetchall()
    conn.close()
    if not rows:
        return "当前没有未执行的定时提醒"
    return "\n".join(
        f"[{rid}] ({repeat} {at}) {content}" for rid, content, at, repeat in rows
    )

@tool
def cancel_reminder(reminder_id: int) -> str:
    """取消一个定时提醒（编号见 list_reminders）。"""
    conn = _db()
    cur = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()
    return f"已取消提醒 #{reminder_id}（{cur.rowcount} 条）" if cur.rowcount else f"提醒 #{reminder_id} 不存在"

def _recent_memos(n: int = 10) -> list:
    """读取桌面 memos.md 最近 n 条备忘。"""
    try:
        if not MEMO_FILE.exists():
            return []
        lines = [
            l.strip().lstrip("- ").strip()
            for l in MEMO_FILE.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        return lines[-n:]
    except OSError:
        return []

def _recent_kb(n: int = 3) -> list:
    """知识库最近沉淀的片段（按保存顺序取末尾 n 条）。"""
    try:
        if not KB_CHUNKS.exists():
            return []
        data = json.loads(KB_CHUNKS.read_text(encoding="utf-8"))
        return [f"{c.get('doc', '文档')}：{c.get('text', '')[:60]}" for c in data[-n:]]
    except Exception:
        return []

@tool
def generate_daily_report(city: str = "十堰") -> str:
    """生成今日聚合日报：自动汇总 日期/城市天气/未完成待办/最近备忘/知识库最近沉淀，并由 AI 写一段今日总结，保存为桌面“日报_日期.md”。
    用户说“生成今日日报”“每日报告”“把今天的情况汇总一下”时调用；city 可指定城市（默认十堰）。"""
    from datetime import date as _date

    today = _date.today()
    wd = "一二三四五六日"[today.weekday()]
    weather = get_weather.invoke({"city": city}) if city else "（未指定城市，跳过天气）"
    todos = list_todos.invoke({})
    memos = _recent_memos()
    kb = _recent_kb()

    lines = [f"# 📅 每日报告 {today.isoformat()} 星期{wd}", ""]
    lines += [f"## ☀️ 天气（{city}）", weather, ""]
    lines += ["## ✅ 待办", todos, ""]
    lines += ["## 📝 最近备忘"]
    lines += [f"- {m}" for m in memos] if memos else ["（暂无）"]
    lines.append("")
    lines += ["## 📚 知识库最近沉淀"]
    lines += [f"- {k}" for k in kb] if kb else ["（暂无）"]
    lines.append("")
    body = "\n".join(lines)

    # AI 今日总结（失败则降级为无总结，不影响日报本身）
    try:
        resp = call_llm(deepseek_model(), 
            [
                SystemMessage(
                    "你是个人助理，根据下面的天气、待办、备忘写 2-3 句简洁的今日总结，"
                    "给出行事建议，不要客套，不要复述全部内容。"
                ),
                HumanMessage(content=body),
            ]
        )
        body += "\n## 💡 今日总结\n" + str(resp.content) + "\n"
    except Exception:
        pass

    filename = f"日报_{today.isoformat()}.md"
    path = Path(os.path.expanduser("~/Desktop")) / filename
    try:
        path.write_text(body, encoding="utf-8")
        saved = str(path)
    except OSError:
        saved = f"桌面写入失败，内容如下：\n{body}"
    return f"已生成今日日报：{saved}\n\n{body}"

@tool
def schedule_daily_report(at: str, city: str = "十堰") -> str:
    """注册每日自动生成日报：每天 at（格式 HH:MM，如 08:00）时刻自动聚合 天气/待办/备忘/知识库 生成日报文件，并弹窗通知。
    用户说“每天早上 X 点生成日报”“定时日报”“每天自动汇总”时调用。"""
    _parse_at(at, "daily")  # 校验 HH:MM
    conn = _db()
    conn.execute(
        "INSERT INTO reminders (content, at, repeat, created_at, action, action_arg) "
        "VALUES (?,?,?,?,?,?)",
        (
            f"每日日报聚合（城市：{city}）",
            at,
            "daily",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "daily_report",
            city,
        ),
    )
    conn.commit()
    conn.close()
    return f"已注册每日日报任务：每天 {at} 自动生成（城市：{city}）"

def _is_due(at: str, repeat: str, now: datetime):
    try:
        dt, wd, hm = _parse_at(at, repeat)
    except Exception:
        return False
    if repeat == "once":
        return dt <= now
    if repeat == "daily":
        return now.strftime("%H:%M") == hm
    if repeat == "weekly":
        return now.weekday() == wd and now.strftime("%H:%M") == hm
    return False

def _advance_at(at: str, repeat: str):
    """触发后推进下次时间。"""
    from datetime import timedelta

    if repeat == "daily":
        return (datetime.now() + timedelta(days=1)).strftime("%H:%M")
    if repeat == "weekly":
        _, wd, hm = _parse_at(at, repeat)
        return f"{'一二三四五六日'[wd]} {hm}"
    return at

def _check_due():
    from datetime import timedelta

    now = datetime.now()
    conn = _db()
    rows = conn.execute(
        "SELECT id, content, at, repeat, action, action_arg FROM reminders"
    ).fetchall()
    for rid, content, at, repeat, action, action_arg in rows:
        if not _is_due(at, repeat, now):
            continue
        fired_at = now.strftime("%Y-%m-%d %H:%M")
        alert_text = f"⏰ {content}"
        if action == "daily_report":
            # 动作型提醒：到点真正执行“聚合日报”任务链
            try:
                report = generate_daily_report(city=action_arg or "")
                head, _, rest = report.partition("\n")
                preview = "\n".join(rest.split("\n")[:3])
                alert_text = f"📅 今日日报已生成\n{head}\n{preview}..."
            except Exception as e:
                alert_text = f"📅 日报生成失败：{e}"
        conn.execute(
            "INSERT INTO alerts (content, fired_at) VALUES (?,?)",
            (alert_text, fired_at),
        )
        try:
            with open(MEMO_FILE, "a", encoding="utf-8") as f:
                f.write(f"- [提醒 {fired_at}] {alert_text}\n")
        except OSError:
            pass
        if repeat == "once":
            conn.execute("DELETE FROM reminders WHERE id = ?", (rid,))
        else:
            conn.execute(
                "UPDATE reminders SET at=? WHERE id=?",
                (_advance_at(at, repeat), rid),
            )
    conn.commit()
    conn.close()

def start_scheduler():
    """启动后台调度线程（幂等）：每 30 秒检查一次到期提醒。"""
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True

    def _loop():
        while True:
            try:
                _check_due()
            except Exception:
                pass
            _time.sleep(30)

    import threading

    t = threading.Thread(target=_loop, daemon=True)
    t.start()

    # 顺带启动本地提醒 HTTP 服务（前端 JS 轮询用，与 Streamlit 页面刷新完全解耦）
    start_alert_http_server()

class _AlertHandler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse

        try:
            u = urlparse(self.path)
            if u.path == "/alerts":
                rows = get_unread_alerts()
                self._send(200, rows)
            elif u.path == "/ack":
                qs = parse_qs(u.query)
                aid = qs.get("id", [""])[0]
                if aid:
                    conn = _db()
                    conn.execute("UPDATE alerts SET shown=1 WHERE id=?", (aid,))
                    conn.commit()
                    conn.close()
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except Exception:
            self._send(500, {"error": "internal"})

    def log_message(self, *args):
        pass

def start_alert_http_server():
    global _alert_server
    if _alert_server:
        return
    import threading
    from http.server import ThreadingHTTPServer

    try:
        _alert_server = ThreadingHTTPServer(("127.0.0.1", _ALERT_HTTP_PORT), _AlertHandler)
        threading.Thread(target=_alert_server.serve_forever, daemon=True).start()
    except OSError as e:
        # 端口被占用说明服务已在运行；否则写日志便于排查
        try:
            Path(__file__).parent.joinpath("_alert_http.log").write_text(
                f"bind failed: {e!r}", encoding="utf-8"
            )
        except Exception:
            pass

def get_unread_alerts() -> list:
    conn = _db()
    rows = conn.execute(
        "SELECT id, content, fired_at FROM alerts WHERE shown=0 ORDER BY id"
    ).fetchall()
    conn.close()
    return [{"id": i, "content": c, "fired_at": t} for i, c, t in rows]

def mark_alerts_shown():
    conn = _db()
    conn.execute("UPDATE alerts SET shown=1 WHERE shown=0")
    conn.commit()
    conn.close()

__all__ = [n for n in list(globals()) if not n.startswith("__")]
