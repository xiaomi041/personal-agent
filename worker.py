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
- GOAL_RESUME=0        关掉目标续跑（只保留提醒服务）
"""
import logging
import os
import signal
import sys
import time

import goals
import scheduler

log = logging.getLogger("worker")

_stop = False

GOAL_TICK = float(os.getenv("GOAL_TICK_SECONDS", "60"))
GOAL_STALE = float(os.getenv("GOAL_STALE_SECONDS", "120"))
GOAL_MAX_ATTEMPTS = int(os.getenv("GOAL_MAX_ATTEMPTS", "2"))
GOAL_RESUME = os.getenv("GOAL_RESUME", "1") != "0"


def _handle(signum, _frame):
    global _stop
    _stop = True
    log.info("收到信号 %s，准备退出", signum)


def resume_goals(limit: int = 2, graph=None) -> dict:
    """扫一遍"停在半路"的目标，把它们接着跑完。

    返回 {"picked": n, "done": n, "failed": n, "left": n}，方便 --once 打印与测试断言。
    """
    rows = goals.stale(GOAL_STALE, limit=limit) if GOAL_RESUME else []
    out = {"picked": len(rows), "done": 0, "failed": 0, "left": 0}
    if not rows:
        return out
    if graph is None:
        import main  # 延迟导入：只跑提醒服务时不必把整张图拉起来

        graph = main.get_graph()
    for g in rows:
        gid = int(g["id"])
        tries = goals.bump_attempt(gid)
        if tries > GOAL_MAX_ATTEMPTS:
            goals.set_status(gid, goals.STATUS_ABORTED,
                             f"断续续跑 {tries - 1} 次仍未完成，已放弃，请人工看一下。")
            out["failed"] += 1
            log.warning("目标 #%d 续跑次数超限，判失败", gid)
            continue
        text = g.get("text") or ""
        log.info("续跑目标 #%d（第 %d 次）：%s", gid, tries, text[:40])
        try:
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
        except Exception as e:  # noqa: BLE001 - 单条目标失败不能拖垮整个 worker
            log.exception("目标 #%d 续跑出错：%s", gid, str(e)[:160])
            out["failed"] += 1
            continue
        status = (goals.get_goal(gid) or {}).get("status")
        if status == goals.STATUS_DONE:
            out["done"] += 1
        elif status == goals.STATUS_ACTIVE:
            out["left"] += 1
    return out


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
    log.info("目标续跑：%s | 扫描 %ss | 判定停在半路 %ss | 单目标最多续跑 %s 次",
             "开启" if GOAL_RESUME else "关闭", GOAL_TICK, GOAL_STALE, GOAL_MAX_ATTEMPTS)

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
            log.info("心跳：进程存活，待触发提醒 %s 条，待推进目标 %s 条", snap, pending)
    log.info("worker 退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
