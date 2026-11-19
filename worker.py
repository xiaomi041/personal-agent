"""
提醒服务 + 目标推进 常驻进程（headless worker）

为什么需要它：
- Streamlit 只在有浏览器会话时才执行脚本；一旦没人打开页面，
  a.py 里启动的调度线程与 8600 提醒 HTTP 服务都不会存在。
- 结果就是：trainAI 盯票命中后无处投递、到点提醒也不会触发。
- 同理，一个"目标"如果跑到一半（进程被杀、页面关掉、模型报错）就停在那儿，
  也需要有人把它捡起来接着跑 —— 这就是下面的目标队列线程。

这个进程把「定时调度 + 提醒 HTTP + 目标续跑」独立出来，在容器里常驻：

    python worker.py              # 常驻
    python worker.py --once       # 只跑一轮目标队列就退出（测试/排障用）

环境变量：
- ALERT_BIND=0.0.0.0   让同网络的 personal-ai / train-ai 能访问（默认 127.0.0.1）
- ALERT_API_URL        前端 a.py 轮询的地址（compose 里指向本服务）
- APP_PASSWORD         POST /remind 需要带同一个 token
- GOAL_TICK_SECONDS    目标队列扫描间隔（默认 60）
- GOAL_STALE_SECONDS   多久没更新算"停在半路"（默认 120）
- GOAL_MAX_ATTEMPTS    同一条目标最多续跑几次（默认 2），超了就判失败
- GOAL_CONCURRENCY     同时推进几个**不同会话**的目标（默认 3；同一会话永远串行）
- GOAL_LEASE_SECONDS   认领租约（默认 180），防止并发扫描重复认领同一条
- GOAL_RESUME=0        关掉目标续跑（只保留提醒服务）
"""
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import goals
import scheduler

log = logging.getLogger("worker")

_stop = False

GOAL_TICK = float(os.getenv("GOAL_TICK_SECONDS", "60"))
GOAL_STALE = float(os.getenv("GOAL_STALE_SECONDS", "120"))
GOAL_MAX_ATTEMPTS = int(os.getenv("GOAL_MAX_ATTEMPTS", "2"))
GOAL_RESUME = os.getenv("GOAL_RESUME", "1") != "0"
# 同时推进几条**互不相干**的目标。同一会话（thread_id）里的目标仍然串行：
# 它们共用一份 LangGraph checkpoint，并发跑会互相覆盖对方的中间状态。
GOAL_CONCURRENCY = max(1, int(os.getenv("GOAL_CONCURRENCY", "3")))
# 认领租约：抢到目标后给自己留一段独占时间，防止别的线程/进程重复认领。
GOAL_LEASE = float(os.getenv("GOAL_LEASE_SECONDS", "180"))


def _handle(signum, _frame):
    global _stop
    _stop = True
    log.info("收到信号 %s，准备退出", signum)


def _run_one(g: dict, graph) -> str:
    """推进单条目标。返回 done / failed / left / skip。

    先 `claim()` 原子认领——查询到目标不等于抢到了目标，并发扫描时这一步才是互斥点。
    """
    gid = int(g["id"])
    if not goals.claim(gid, GOAL_LEASE):
        log.info("目标 #%d 已被其他线程认领，本轮跳过", gid)
        return "skip"
    try:
        tries = goals.bump_attempt(gid)
        if tries > GOAL_MAX_ATTEMPTS:
            goals.set_status(gid, goals.STATUS_ABORTED,
                             f"断续续跑 {tries - 1} 次仍未完成，已放弃，请人工看一下。")
            log.warning("目标 #%d 续跑次数超限，判失败", gid)
            return "failed"
        text = g.get("text") or ""
        log.info("续跑目标 #%d（第 %d 次，会话 %s）：%s",
                 gid, tries, g.get("thread_id") or "user-main", text[:40])
        from langchain_core.messages import HumanMessage

        graph.invoke(
            {
                "messages": [HumanMessage(content=text)],
                "resume": True,
                # 带上 goal_id/checks：续跑必须写回**同一条**目标记录，
                # 否则规划节点会以为这是新委托、新建一条，原目标永远停在 active。
                "goal_id": gid,
                "checks": list(g.get("checks") or []),
            },
            config={"configurable": {"thread_id": g.get("thread_id") or "user-main"}},
        )
        status = (goals.get_goal(gid) or {}).get("status")
        if status == goals.STATUS_DONE:
            _notify_done(gid, text)
            return "done"
        if status == goals.STATUS_ACTIVE:
            return "left"
        return "left"
    except Exception as e:  # noqa: BLE001 - 单条目标失败不能拖垮整个 worker
        log.exception("目标 #%d 续跑出错：%s", gid, str(e)[:160])
        return "failed"
    finally:
        goals.release(gid)


def _group_by_thread(rows: list) -> list:
    """按 thread_id 分组，组内保持 id 升序；组间按最早一条的 id 排序。"""
    groups: dict = {}
    for g in rows:
        groups.setdefault(g.get("thread_id") or "user-main", []).append(g)
    return sorted(groups.values(), key=lambda gs: int(gs[0]["id"]))


def resume_goals(limit: int | None = None, graph=None, concurrency: int | None = None) -> dict:
    """扫一遍"停在半路"的目标，把它们接着跑完。

    并发策略：**按会话分组、组间并行、组内串行**。
    - 组间并行：不同 thread 的 checkpoint 互不相干，可以同时推进（GOAL_CONCURRENCY 封顶）；
    - 组内串行：同一 thread 共用一个 checkpoint，并发跑等于互相踩，必须排队；
    - 每条先 `claim()` 抢占，抢不到就跳过（多进程/重复扫描都安全）。

    返回 {"picked": n, "done": n, "failed": n, "left": n, "skipped": n, "groups": n}。
    """
    # 每轮捞多少条：默认等于并发度，否则并发再多也只有一个并发度的活可干。
    width_cap = max(1, int(concurrency or GOAL_CONCURRENCY))
    if limit is None:
        limit = width_cap
    rows = goals.stale(GOAL_STALE, limit=limit) if GOAL_RESUME else []
    out = {"picked": len(rows), "done": 0, "failed": 0, "left": 0,
           "skipped": 0, "groups": 0}
    if not rows:
        return out
    if graph is None:
        import main  # 延迟导入：只跑提醒服务时不必把整张图拉起来

        graph = main.get_graph()

    groups = _group_by_thread(rows)
    out["groups"] = len(groups)
    width = max(1, min(int(concurrency or GOAL_CONCURRENCY), len(groups)))
    tally: dict = {"done": 0, "failed": 0, "left": 0, "skipped": 0}
    guard = threading.Lock()

    def run_group(gs: list) -> None:
        for g in gs:
            res = _run_one(g, graph)
            with guard:
                tally[res] = tally.get(res, 0) + 1

    if width == 1:
        for gs in groups:
            run_group(gs)
    else:
        log.info("并发推进：%d 条目标 / %d 个会话，并发度 %d", len(rows), len(groups), width)
        with ThreadPoolExecutor(max_workers=width, thread_name_prefix="goal") as pool:
            list(pool.map(run_group, groups))

    out.update(tally)
    return out



def _notify_done(gid: int, text: str) -> None:
    """目标跑完写一条提醒弹窗 —— 用户不必盯着页面等结果。"""
    g = goals.get_goal(gid) or {}
    result = (g.get("result") or "").strip().replace("\n", " ")
    head = f"✅ 目标完成｜{text[:40]}"
    if result:
        head += f"\n{result[:120]}"
    try:
        scheduler.push_alert(head)
    except Exception as e:  # noqa: BLE001 - 通知失败不能影响续跑结果
        log.warning("目标 #%d 完成，但写提醒失败：%r", gid, e)


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        force=True,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if "--once" in sys.argv:
        got = resume_goals()
        log.info("单轮目标推进：%s", got)
        return 0

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    # start_scheduler 内部会拉起 30s 轮询线程 + 8600 HTTP 服务（幂等）
    scheduler.start_scheduler()
    log.info("提醒 worker 已启动：轮询 30s，HTTP 监听 %s:8600",
             os.getenv("ALERT_BIND", "127.0.0.1"))
    log.info("目标续跑：%s | 扫描 %ss | 判定停在半路 %ss | 单目标最多续跑 %s 次 | 并发 %s 路",
             "开启" if GOAL_RESUME else "关闭", GOAL_TICK, GOAL_STALE,
             GOAL_MAX_ATTEMPTS, GOAL_CONCURRENCY)

    last_beat = 0.0
    last_goal = 0.0
    while not _stop:
        time.sleep(1)
        now = time.monotonic()
        if GOAL_RESUME and now - last_goal >= GOAL_TICK:
            last_goal = now
            try:
                got = resume_goals()
                if got["picked"]:
                    log.info("目标推进一轮：%s", got)
            except Exception as e:  # noqa: BLE001
                log.exception("目标推进线程出错：%s", str(e)[:160])
        if now - last_beat >= 300:
            last_beat = now
            snap = scheduler.pending_reminders_count() if hasattr(
                scheduler, "pending_reminders_count") else -1
            pending = len(goals.pending(50)) if GOAL_RESUME else -1
            log.info("心跳：进程存活，待触发提醒 %s 条，待推进目标 %s 条，并发上限 %s",
                     snap, pending, GOAL_CONCURRENCY)
    log.info("worker 退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
