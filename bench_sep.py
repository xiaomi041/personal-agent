"""检索区分度：可答题 vs 不可答题的分数间隔（纯向量 vs 混合）。

比 recall 更能说明问题：如果不可答题的 top-1 分数也很高，
阈值就拦不住「硬答」，系统就会编。
"""
from __future__ import annotations

import numpy as np

import bench_ab as B
import eval_docs
from knowledge import _bm25_scores, _embed_texts


def top1_scores(chunks, vecs, q: str, mode: str) -> float:
    qv = np.array(_embed_texts([q])[0], dtype="float32")
    qn = np.linalg.norm(qv) + 1e-9
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).ravel() + 1e-9
    cosine = (vecs @ qv) / (norms * qn)
    if mode == "vector":
        return float(cosine.max())
    bm25 = _bm25_scores(q, [c["text"] for c in chunks])
    return float((0.7 * cosine + 0.3 * bm25).max())


def run() -> None:
    B.prep()
    chunks, vecs = B._chunks_of_thread()
    ans, unans = [], []
    for q, _kw, _d, answerable in eval_docs.QA:
        row = (q, *[top1_scores(chunks, vecs, q, m) for m in ("vector", "hybrid")])
        (ans if answerable else unans).append(row)

    print("\n## D. 可答 / 不可答 的 top-1 分数分布（阈值 0.42）")
    print("| 类别 | 纯向量 最低–最高 | 混合 最低–最高 |")
    print("|---|---|---|")
    for name, rows in (("可答 (13)", ans), ("不可答 (4)", unans)):
        v = [r[1] for r in rows]
        h = [r[2] for r in rows]
        print(f"| {name} | {min(v):.3f}–{max(v):.3f} | {min(h):.3f}–{max(h):.3f} |")

    for mode, i in (("vector", 1), ("hybrid", 2)):
        gap = min(r[i] for r in ans) - max(r[i] for r in unans)
        print(f"- {mode}: 可答最低 {min(r[i] for r in ans):.3f} − 不可答最高 "
              f"{max(r[i] for r in unans):.3f} = 间隔 {gap:+.3f}")

    print("\n### 逐题（不可答题最容易暴露问题）")
    print("| 问题 | 可答 | 纯向量 | 混合 |")
    print("|---|---|---:|---:|")
    for q, v, h in ans + unans:
        flag = "是" if any(r[0] == q for r in ans) else "**否**"
        print(f"| {q} | {flag} | {v:.3f} | {h:.3f} |")


if __name__ == "__main__":
    run()
