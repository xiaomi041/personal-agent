"""
提醒服务常驻进程（headless worker）

为什么需要它：
- Streamlit 只在有浏览器会话时才执行脚本；一旦没人打开页面，
  a.py 里启动的调度线程与 8600 提醒 HTTP 服务都不会存在。
- 结果就是：trainAI 盯票命中后无处投递、到点提醒也不会触发。

这个进程把「定时调度 + 提醒 HTTP」独立出来，在容器里常驻：

    python worker.py

环境变量：
- ALERT_BIND=0.0.0.0  让同网络的 personal-ai / train-ai 能访问（默认 127.0.0.1）
- ALERT_API_URL       前端 a.py 轮询的地址（compose 里指向本服务）
- APP_PASSWORD        POST /remind 需要带同一个 token
"""
import logging
import os
import signal
import time

import scheduler

log = logging.getLogger("worker")

_stop = False


def _handle(signum, _frame):
    global _stop
    _stop = True
    log.info("收到信号 %s，准备退出", signum)


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        force=True,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    # start_scheduler 内部会拉起 30s 轮询线程 + 8600 HTTP 服务（幂等）
    scheduler.start_scheduler()
    log.info("提醒 worker 已启动：轮询 30s，HTTP 监听 %s:8600",
             os.getenv("ALERT_BIND", "127.0.0.1"))

    last_beat = 0.0
    while not _stop:
        time.sleep(1)
        now = time.monotonic()
        if now - last_beat >= 300:
            last_beat = now
            snap = scheduler.pending_reminders_count() if hasattr(
                scheduler, "pending_reminders_count") else -1
            log.info("心跳：进程存活，待触发提醒 %s 条", snap)
    log.info("worker 退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
