"""对照实验：混合检索到底比纯向量强多少？P99 是多少？上下文涨多少？

三组实验，全部在容器内跑：
  A. 检索延迟 P50/P95/P99（100 次查询）
  B. 纯向量 vs 混合检索 的 recall@1 / recall@3（同一批 17 条问题）
  C. 上下文增长：同一问题在 0 / 3 / 6 轮历史后的输入 token

用法：docker compose exec -T personal-ai python bench_ab.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import eval_docs
import main
from knowledge import _bm25_scores, _embed_texts, _load_kb

THREAD = os.getenv("AB_THREAD", "ab-thread")
OUT = []


def say(line: str = "") -> None:
    print(line, flush=True)
    OUT.append(line)


# ------------------------------------------------------------ 语料
def prep() -> None:
    main.bind_thread(THREAD)
    for fname, builder in (
        ("合作协议.pdf", eval_docs.build_contract_pdf),
        ("产品手册.docx", eval_docs.build_manual_docx),
        ("合同扫描件.png", eval_docs.build_scan_png),
        ("长文档.pdf", eval_docs.build_long_pdf),
    ):
        main.delete_document(fname, thread_id=THREAD)
        t0 = time.time()
        main.add_document(fname, builder(), thread_id=THREAD)
        say(f"[入库] {fname} {int((time.time() - t0) * 1000)} ms")


def _chunks_of_thread() -> tuple[list, np.ndarray]:
    chunks, vecs, _ = _load_kb()
    keep = [i for i, c in enumerate(chunks) if c["thread_id"] == THREAD]
    return [chunks[i] for i in keep], vecs[keep]


# ------------------------------------------------------------ A. 延迟分位
def part_a(chunks, vecs, rounds: int = 10) -> None:
    queries = [q for q, *_ in eval_docs.QA]
    lat, scored = [], []
    for _ in range(rounds):
        for q in queries:
            t0 = time.time()
            got = main.search_documents(q, thread_id=THREAD)
            lat.append((time.time() - t0) * 1000)
            scored.append(bool(got))
    s = sorted(lat)

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))]

    say(f"\n## A. 检索延迟（{len(lat)} 次：{len(queries)} 条问题 × {rounds} 轮）")
    say(f"- P50 {q(50):.0f} ms / P95 {q(95):.0f} ms / P99 {q(99):.0f} ms / 最大 {s[-1]:.0f} ms")
    say(f"- 有结果返回 {sum(scored)}/{len(scored)}（阈值 {os.getenv('DOC_MIN_SCORE', '0.42')} 过滤）")


# ------------------------------------------------------------ B. 纯向量 vs 混合
def _rank(q: str, chunks, vecs, mode: str, top_k: int = 3) -> list[dict]:
    qv = np.array(_embed_texts([q])[0], dtype="float32")
    qn = np.linalg.norm(qv) + 1e-9
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).ravel() + 1e-9
    cosine = (vecs @ qv) / (norms * qn)
    if mode == "vector":
        score = cosine
    else:
        bm25 = _bm25_scores(q, [c["text"] for c in chunks])
        score = 0.7 * cosine + 0.3 * bm25
    idx = np.argsort(-score)[:top_k]
    return [dict(chunks[i], _score=float(score[i])) for i in idx]


def _hit(kw: str | tuple | None, items: list[dict]) -> bool:
    if kw is None or not items:
        return False
    cands = kw if isinstance(kw, tuple) else (kw,)
    return any(c in it["text"] for it in items for c in cands)


def part_b(chunks, vecs) -> None:
    say("\n## B. 纯向量 vs 混合检索（同一批问题，同一份语料）")
    say("| 方案 | recall@1 | recall@3 | 平均检索耗时 |")
    say("|---|---:|---:|---:|")
    stats = {}
    for mode in ("vector", "hybrid"):
        r1 = r3 = 0
        elapsed = []
        for q, kw, _doc_kw, answerable in eval_docs.QA:
            if not answerable:
                continue
            t0 = time.time()
            top3 = _rank(q, chunks, vecs, mode, 3)
            elapsed.append((time.time() - t0) * 1000)
            r1 += _hit(kw, top3[:1])
            r3 += _hit(kw, top3)
        n = sum(1 for *_, a in eval_docs.QA if a)
        stats[mode] = (r1, r3, n, sum(elapsed) / len(elapsed))
        label = "纯向量（cosine）" if mode == "vector" else "混合（0.7 余弦 + 0.3 BM25）"
        say(f"| {label} | {r1}/{n} | {r3}/{n} | {stats[mode][3]:.0f} ms |")

    d1 = stats["hybrid"][0] - stats["vector"][0]
    d3 = stats["hybrid"][1] - stats["vector"][1]
    say(f"\n- recall@1 变化：{d1:+d} 条；recall@3 变化：{d3:+d} 条（共 {stats['vector'][2]} 条可答题）")
    if stats["vector"][2]:
        say(f"- 相对提升：recall@1 {d1 / stats['vector'][2] * 100:+.1f}pp，"
            f"recall@3 {d3 / stats['vector'][2] * 100:+.1f}pp")


# ------------------------------------------------------------ C. 上下文增长
def part_c() -> None:
    from langchain_core.messages import HumanMessage

    say("\n## C. 上下文增长（同一问题，前面垫不同轮数的历史）")
    say("| 历史轮数 | 第 1 次调用输入 token | 整轮总输入 token | 耗时 |")
    say("|---|---:|---:|---:|")
    graph = main.get_graph()
    for turns in (0, 3, 6):
        tid = f"ctx-{turns}"
        for i in range(turns):
            graph.invoke(
                {"messages": [HumanMessage(content=f"随便聊一句接口测试内容 {i}，说明一下进度")]},
                config={"configurable": {"thread_id": tid}},
            )
        before = main.usage_snapshot()
        t0 = time.time()
        graph.invoke(
            {"messages": [HumanMessage(content="现在几点了")]},
            config={"configurable": {"thread_id": tid}},
        )
        after = main.usage_snapshot()
        # 用总 token / 调用次数近似「首轮输入」，更直观的是整轮总输入
        used = after["in"] - before["in"]
        calls = max(1, after["calls"] - before["calls"])
        say(f"| {turns} 轮 | {used / calls:.0f} | {used} | {time.time() - t0:.1f}s |")


def run() -> None:
    say("=== 准备语料 ===")
    prep()
    chunks, vecs = _chunks_of_thread()
    say(f"语料：{len(chunks)} 个切片，{len(vecs)} 个向量")
    part_a(chunks, vecs)
    part_b(chunks, vecs)
    part_c()
    with open("bench_ab.json", "w", encoding="utf-8") as f:
        json.dump({"log": OUT}, f, ensure_ascii=False, indent=2)
    say("\n结果已写入 bench_ab.json")


if __name__ == "__main__":
    run()
