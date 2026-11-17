"""上下文实验（控制变量版）：历史到底花多少钱、又值多少钱。

设计：
  Part 1  固定长度历史（用户/助手各一段等长文本）× N 轮，问同一个问题，
          看该轮输入 token 怎么长 —— 历史长度是唯一变量。
  Part 2  同一个追问「违约金 5%，一共多少」：
          A 组带着「合同总额 20 万」的历史 → 能算
          B 组清空历史                    → 算不出
          用来说明"截断历史"的代价：省 token 的同时会丢掉多轮依赖。

用法：docker compose exec -T personal-ai python bench_ctx.py
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import main

# 等长历史：用户句 40 字上下，助手句 30 字上下，长度几乎一致，避免长短干扰
USER_FILLER = "记录一条背景信息：本期项目处于联调阶段，暂无异常，编号 000。"
AI_FILLER = "收到，已记录该背景信息，继续等待下一步安排即可。"


def _invoke(graph, tid: str, text: str) -> tuple[dict, float]:
    before = main.usage_snapshot()
    r = graph.invoke({"messages": [HumanMessage(content=text)]},
                     config={"configurable": {"thread_id": tid}})
    after = main.usage_snapshot()
    delta = {k: after[k] - before[k] for k in ("in", "out", "calls")}
    return delta, r


def part1(graph) -> None:
    print("\n## Part 1. 固定长度历史下的成本增长")
    print("| 历史轮数 | 本轮输入 token | 输出 token | 模型调用 |")
    print("|---|---:|---:|---:|")
    rows = []
    for n in (0, 2, 4, 6):
        tid = f"ctxfix-{n}"
        if n:
            msgs = []
            for i in range(n):
                msgs += [HumanMessage(content=USER_FILLER),
                         AIMessage(content=AI_FILLER)]
            graph.update_state({"configurable": {"thread_id": tid}}, {"messages": msgs})
        d, _ = _invoke(graph, tid, "现在几点了")
        rows.append((n, d["in"], d["out"], d["calls"]))
        print(f"| {n} 轮 | {d['in']} | {d['out']} | {d['calls']} |")

    if len(rows) >= 2:
        per_turn = (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0])
        print(f"\n- 每轮历史约 +{per_turn:.0f} token（等长填充，n={len(rows)} 个点）")


def part2(graph) -> None:
    print("\n## Part 2. 历史的「价值」：多轮依赖能不能算出来")
    question = "按违约金 5% 算，一共要赔多少钱"
    setup = "我的合同总额是 20 万元，记一下"

    # A 组：带着历史
    tid_a = "ctx-with"
    graph.invoke({"messages": [HumanMessage(content=setup)]},
                 config={"configurable": {"thread_id": tid_a}})
    d_a, r_a = _invoke(graph, tid_a, question)
    ans_a = (r_a.get("execution_result") or "")[:90].replace("\n", " ")

    # B 组：同一问题，但没有历史
    tid_b = "ctx-without"
    d_b, r_b = _invoke(graph, tid_b, question)
    ans_b = (r_b.get("execution_result") or "")[:90].replace("\n", " ")

    print(f"- 有历史（{d_a['in']} token）：{ans_a}")
    print(f"- 无历史（{d_b['in']} token）：{ans_b}")
    ok_a = "1" in ans_a or "万" in ans_a
    ok_b = "1" in ans_b or "万" in ans_b
    print(f"\n- 结论：有历史 {'答出' if ok_a else '没答出'}，无历史 {'答出' if ok_b else '没答出'}；"
          f"代价是多花约 {d_a['in'] - d_b['in']} token")


def run() -> None:
    graph = main.get_graph()
    part1(graph)
    part2(graph)


if __name__ == "__main__":
    run()
