#!/usr/bin/env python3
"""token 成本曲线：会话越长，每轮越贵。

做法：往同一会话里预置 N 轮**等长**历史，再问同一个问题，读 usage 计数。
只看一件事——「第几轮之后成本开始失控」，以及换算成人民币是几毛钱。

用法（容器内）：
    docker compose exec -T personal-ai python bench_cost.py
    COST_ROUNDS=0,2,4,6,10,20,30 python bench_cost.py
"""

from __future__ import annotations

import os
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage

import main

ROUNDS = [int(x) for x in os.getenv("COST_ROUNDS", "0,2,4,6,10,20,30").split(",") if x.strip()]
QUESTION = "现在几点了？"

# 等长填充：用户 30 字、助手 24 字，保证每轮增量一致
USER_FILLER = "记录一条背景信息：本期项目处于联调阶段，暂无异常，编号 000。"
AI_FILLER = "收到，已记录该背景信息，继续等待下一步安排即可。"

# DeepSeek 官方价（元 / 百万 token），可用环境变量覆盖
PRICE_IN = float(os.getenv("PRICE_IN", "2.0"))
PRICE_OUT = float(os.getenv("PRICE_OUT", "8.0"))


def _ask(rounds: int) -> dict:
    tid = f"cost-{uuid.uuid4().hex[:8]}"
    main.bind_thread(tid)
    graph = main.get_graph()

    if rounds:
        msgs = []
        for i in range(rounds):
            msgs += [
                HumanMessage(content=USER_FILLER.replace("000", f"{i:03d}")),
                AIMessage(content=AI_FILLER),
            ]
        graph.update_state({"configurable": {"thread_id": tid}}, {"messages": msgs})

    before = main.usage_snapshot()
    started = time.perf_counter()
    graph.invoke(
        {"messages": [("user", QUESTION)]},
        config={"configurable": {"thread_id": tid}},
    )
    elapsed = time.perf_counter() - started
    after = main.usage_snapshot()

    return {
        "rounds": rounds,
        "in": after["in"] - before["in"],
        "out": after["out"] - before["out"],
        "calls": after["calls"] - before["calls"],
        "seconds": elapsed,
    }


def main_() -> None:
    print("token 成本曲线（同一会话里预置等长历史后问同一个问题）")
    print(f"单价假设：输入 ¥{PRICE_IN}/M token，输出 ¥{PRICE_OUT}/M token\n")
    print("| 历史轮数 | 输入 token | 输出 token | 调用次数 | 耗时 s | 单轮成本 ¥ | 相对空会话 |")
    print("|---|---|---|---|---|---|---|")

    first = None
    for rounds in ROUNDS:
        row = _ask(rounds)
        cost = row["in"] / 1e6 * PRICE_IN + row["out"] / 1e6 * PRICE_OUT
        if first is None:
            first = row["in"] or 1
        ratio = row["in"] / first
        print(
            f"| {rounds} | {row['in']} | {row['out']} | {row['calls']} | "
            f"{row['seconds']:.2f} | {cost:.4f} | {ratio:.2f}× |"
        )


if __name__ == "__main__":
    main_()
