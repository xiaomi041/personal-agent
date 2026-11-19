# -*- coding: utf-8 -*-
"""目标对象化与目标队列。

一次委托不再只是"一段对话"，而是一条可判定的目标记录：

    goals(id, thread_id, text, plan, checks, status, result, replans, created_at, updated_at)

- `text`   用户原话（目标本身）
- `plan`   规划节点的计划文本
- `checks` 由规划节点给出的、可判定的完成判据（JSON 数组）
- `status` active / done / aborted / asked

用途有两个：
1. 评审节点按 checks 逐条判定，而不是凭"感觉通过"；
2. watch 式的后台推进（worker）能扫到 `active` 的目标继续做，不必等用户再打字。

为什么单独一个库文件：LangGraph 的 checkpoint 库只放对话状态，
目标队列是"业务状态"，混在一起以后不好备份、也不好单独清理。
"""
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from settings import *  # noqa: F401,F403

log = logging.getLogger("agent.goals")

GOALS_DB = Path(os.getenv("GOALS_DB", str(DATA_DIR / "goals.db")))

_lock = threading.Lock()

STATUS_ACTIVE = "active"
STATUS_DONE = "done"
STATUS_ABORTED = "aborted"
STATUS_ASKED = "asked"

# 允许写入的状态集合：写错状态名直接报错，避免数据库里出现"完成丁"这种脏值。
_STATUSES = {STATUS_ACTIVE, STATUS_DONE, STATUS_ABORTED, STATUS_ASKED}


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(GOALS_DB), check_same_thread=False, timeout=5)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    """建表（幂等）。每个进程第一次用到目标队列时调一次即可。"""
    with _lock:
        with _conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS goals (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id  TEXT NOT NULL DEFAULT '',
                    text       TEXT NOT NULL DEFAULT '',
                    plan       TEXT NOT NULL DEFAULT '',
                    checks     TEXT NOT NULL DEFAULT '[]',
                    status     TEXT NOT NULL DEFAULT 'active',
                    result     TEXT NOT NULL DEFAULT '',
                    replans    INTEGER NOT NULL DEFAULT 0,
                    attempts   INTEGER NOT NULL DEFAULT 0,
                    claim_until REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            # 老库没有 attempts / claim_until 列（2026-11-20 之前建的），补上；
            # 重复 ALTER 会报错，用 PRAGMA 先查。
            cols = {r["name"] for r in c.execute("PRAGMA table_info(goals)").fetchall()}
            if "attempts" not in cols:
                c.execute("ALTER TABLE goals ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
            if "claim_until" not in cols:
                c.execute("ALTER TABLE goals ADD COLUMN claim_until REAL NOT NULL DEFAULT 0")
            c.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status, id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_goals_thread ON goals(thread_id, id)")


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = {k: r[k] for k in r.keys()}
    try:
        d["checks"] = json.loads(d.get("checks") or "[]")
    except (TypeError, ValueError):
        d["checks"] = []
    return d


def add_goal(thread_id: str, text: str, plan: str = "", checks: list | None = None) -> int:
    """新建一条目标，返回 id。checks 为空也允许（老行为：只靠评审结论）。"""
    init()
    now = time.time()
    with _lock:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO goals(thread_id, text, plan, checks, status, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (thread_id or "", text or "", plan or "",
                 json.dumps(list(checks or []), ensure_ascii=False),
                 STATUS_ACTIVE, now, now),
            )
            gid = int(cur.lastrowid)
    log.info("目标已创建 | #%d | %s | 判据 %d 条", gid, (text or "")[:40], len(checks or []))
    return gid


def get_goal(goal_id: int) -> dict | None:
    init()
    with _conn() as c:
        return _row(c.execute("SELECT * FROM goals WHERE id=?", (int(goal_id),)).fetchone())


def set_status(goal_id: int, status: str, result: str | None = None) -> None:
    """更新状态；状态名非法直接抛 KeyError（调用方 bug 要立刻暴露）。"""
    if status not in _STATUSES:
        raise KeyError(f"未知目标状态：{status}")
    init()
    with _lock:
        with _conn() as c:
            if result is None:
                c.execute("UPDATE goals SET status=?, updated_at=? WHERE id=?",
                          (status, time.time(), int(goal_id)))
            else:
                c.execute("UPDATE goals SET status=?, result=?, updated_at=? WHERE id=?",
                          (status, result, time.time(), int(goal_id)))
    log.info("目标状态 | #%d -> %s", int(goal_id), status)


def update_plan(goal_id: int, plan: str, checks: list | None = None) -> None:
    """重规划后替换计划与判据（目标本身不变，变的只是怎么做）。"""
    init()
    with _lock:
        with _conn() as c:
            c.execute("UPDATE goals SET plan=?, checks=?, updated_at=? WHERE id=?",
                      (plan or "", json.dumps(list(checks or []), ensure_ascii=False),
                       time.time(), int(goal_id)))
    log.info("目标重规划 | #%d | 判据 %d 条", int(goal_id), len(checks or []))


def bump_replan(goal_id: int) -> int:
    """重规划次数 +1，返回新值。用于给"计划可以被推翻"封顶。"""
    init()
    with _lock:
        with _conn() as c:
            c.execute("UPDATE goals SET replans=replans+1, updated_at=? WHERE id=?",
                      (time.time(), int(goal_id)))
            row = c.execute("SELECT replans FROM goals WHERE id=?", (int(goal_id),)).fetchone()
    return int(row["replans"]) if row else 0


def list_goals(status: str | None = None, thread_id: str | None = None, limit: int = 50) -> list:
    init()
    sql = "SELECT * FROM goals"
    where, args = [], []
    if status:
        where.append("status=?")
        args.append(status)
    if thread_id:
        where.append("thread_id=?")
        args.append(thread_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        return [_row(r) for r in c.execute(sql, args).fetchall()]


def pending(limit: int = 5) -> list:
    """后台推进要处理的目标：只取 active，按创建时间先来先做。"""
    init()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM goals WHERE status=? ORDER BY id ASC LIMIT ?",
            (STATUS_ACTIVE, int(limit)),
        ).fetchall()
    return [_row(r) for r in rows]


def stale(seconds: float, limit: int = 3, include_claimed: bool = False) -> list:
    """超过 `seconds` 秒没有任何更新的 active 目标 —— 判定为"停在半路"，可以续跑。

    这里是个近似判断：目标在执行过程中不会刷新 `updated_at`，所以长任务也会被捞出来。
    多线程/多进程同时推进时还会互相抢同一条，所以真正认领要让 `claim()` 原子地抢；
    这里默认把"已被别人认领且租约未到期"的目标排除掉。
    """
    init()
    cutoff = time.time() - max(1.0, float(seconds))
    sql = "SELECT * FROM goals WHERE status=? AND updated_at<=?"
    args = [STATUS_ACTIVE, cutoff]
    if not include_claimed:
        sql += " AND claim_until<=?"
        args.append(time.time())
    sql += " ORDER BY id ASC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        rows = c.execute(sql, args).fetchall()
    return [_row(r) for r in rows]


def claim(goal_id: int, lease: float = 180.0) -> bool:
    """原子认领一条目标：只有"还没被认领 / 租约已过期"的才能抢到。

    这是多线程推进的关键——`stale()` 只是查询，两个线程同时查到同一条会各跑一遍，
    既浪费 token 也会互相覆盖结果。真正的互斥靠这条 UPDATE 的 WHERE 条件。
    """
    init()
    now = time.time()
    with _lock:
        with _conn() as c:
            cur = c.execute(
                "UPDATE goals SET claim_until=? WHERE id=? AND status=? AND claim_until<=?",
                (now + max(5.0, float(lease)), int(goal_id), STATUS_ACTIVE, now),
            )
            return cur.rowcount == 1


def release(goal_id: int) -> None:
    """放掉认领（跑完/失败都要放，否则租约到期前没人能再捡）。"""
    init()
    with _lock:
        with _conn() as c:
            c.execute("UPDATE goals SET claim_until=0 WHERE id=?", (int(goal_id),))



def bump_attempt(goal_id: int) -> int:
    """续跑次数 +1（封顶由调用方控制），返回新值。"""
    init()
    with _lock:
        with _conn() as c:
            c.execute("UPDATE goals SET attempts=attempts+1, updated_at=? WHERE id=?",
                      (time.time(), int(goal_id)))
            row = c.execute("SELECT attempts FROM goals WHERE id=?", (int(goal_id),)).fetchone()
    return int(row["attempts"]) if row else 0


def stats(days: int = 7) -> dict:
    """目标队列的简单统计，给界面和巡检脚本用。"""
    init()
    since = time.time() - max(1, int(days)) * 86400
    with _conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM goals WHERE created_at>=? GROUP BY status",
                         (since,)).fetchall()
        total = c.execute("SELECT COUNT(*) n FROM goals WHERE created_at>=?", (since,)).fetchone()["n"]
        replans = c.execute("SELECT COALESCE(SUM(replans),0) s FROM goals WHERE created_at>=?",
                            (since,)).fetchone()["s"]
    counts = {r["status"]: r["n"] for r in rows}
    done = counts.get(STATUS_DONE, 0)
    return {
        "days": days,
        "total": int(total),
        "done": int(done),
        "aborted": int(counts.get(STATUS_ABORTED, 0)),
        "asked": int(counts.get(STATUS_ASKED, 0)),
        "active": int(counts.get(STATUS_ACTIVE, 0)),
        "replans": int(replans or 0),
        "success_rate": round(done / total, 3) if total else 0.0,
    }


def reset() -> None:
    """清空目标表（测试与本地调试用）。"""
    init()
    with _lock:
        with _conn() as c:
            c.execute("DELETE FROM goals")


__all__ = [n for n in list(globals()) if not n.startswith("__")]
