#!/usr/bin/env python3
"""并发曲线：同一实例上并行跑 N 个简单委托，看墙钟、P50/P95 与吞吐。

只做一件事：把「4 路并发不串数据」这句结论，换成一条 1 → 2 → 4 → 8 的曲线。
每个请求用独立 thread_id，避免互相看到对方的历史。

用法（容器内）：
    docker compose exec -T personal-ai python bench_conc.py
    CONC_LEVELS=1,2,4 CONC_REQUESTS=8 python bench_conc.py
"""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import main

QUESTIONS = [
    "现在几点了？",
    "今天几号？",
    "看一下我的待办",
    "现在几点了？",
]

LEVELS = [int(x) for x in os.getenv("CONC_LEVELS", "1,2,4,8").split(",") if x.strip()]
REQUESTS = int(os.getenv("CONC_REQUESTS", "8"))


def _one(index: int) -> tuple[float, bool]:
    """跑一条委托，返回 (耗时秒, 是否成功)。"""
    tid = f"conc-{uuid.uuid4().hex[:8]}"
    main.bind_thread(tid)
    started = time.perf_counter()
    try:
        main.get_graph().invoke(
            {"messages": [("user", QUESTIONS[index % len(QUESTIONS)])]},
            config={"configurable": {"thread_id": tid}},
        )
        ok = True
    except Exception as exc:  # noqa: BLE001 - 压测要记录失败而不是中断
        print(f"  请求 {index} 失败：{type(exc).__name__}: {str(exc)[:120]}")
        ok = False
    return time.perf_counter() - started, ok


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[idx]


def main_() -> None:
    print("并发曲线（每条委托都是真实模型调用）")
    print(f"档位：{LEVELS}；每档 {REQUESTS} 条请求\n")
    print("| 并发 | 墙钟 s | 吞吐 req/s | P50 s | P95 s | 成功 |")
    print("|---|---|---|---|---|---|")
    for level in LEVELS:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            results = list(pool.map(_one, range(REQUESTS)))
        wall = time.perf_counter() - started
        times = [t for t, ok in results if ok]
        good = sum(1 for _, ok in results if ok)
        print(
            f"| {level} | {wall:.2f} | {REQUESTS / wall:.2f} | "
            f"{_pct(times, 0.5):.2f} | {_pct(times, 0.95):.2f} | {good}/{REQUESTS} |"
        )


if __name__ == "__main__":
    main_()
