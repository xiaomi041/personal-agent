"""难度更高的检索对照：同一批问题，三种权重下的命中率。

为什么要另开一份：`bench_tasks.py` 的 40 条任务在 BM25 单路下也能全过，
说明那些问法和原文关键词重合太多，测不出「混合检索到底有没有用」。

这里换一种问法——**只描述意图，不复述原文词**（「交付延期要赔多少钱」而不是
「违约金比例是多少」），再分别用三种权重跑同一批题：

- 纯向量  (1.0 / 0.0)
- 纯关键词 (0.0 / 1.0)
- 混合    (0.7 / 0.3，产品默认)

判定只看「命中的片段属于不在正确的文档」，不看模型怎么回答——这是检索层评测。

运行：python bench_hard.py
"""
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DATA_DIR", os.path.join(HERE, ".tmp-hard"))
os.environ["DOC_MIN_SCORE"] = "0.05"  # 只排序，不拒绝，便于比较三种模式的相对好坏
sys.path.insert(0, HERE)

THREAD = os.getenv("HARD_THREAD") or f"hard-bench-{time.strftime('%m%d-%H%M%S')}"

# (问题, 期望文档, 期望出现在片段里的关键词之一)
CASES = [
    ("如果我方交付延期，要赔给对方多少钱？", "合作协议.pdf", ("违约金", "5%")),
    ("对方不守约定时，我们找哪个机构来评理？", "合作协议.pdf", ("仲裁",)),
    ("合作中接触到的内部消息，我们得守口如瓶多久？", "合作协议.pdf", ("保密",)),
    ("我们给对方留的那笔信用保证，金额是多少？", "合作协议.pdf", ("保证金", "10000")),
    ("这份合作文件用什么编号归档？", "合同扫描件.png", ("A-2024-001",)),
    ("这个系统回答一个问题，最多能让人等多久？", "产品手册.docx", ("响应时间", "2s")),
    ("用户找资料时，多大比例要能一次就找对？", "产品手册.docx", ("命中率", "85")),
    ("整个工程对外发布的日子定在哪天？", "长文档.pdf", ("2026", "9 月 30")),
    ("这个项目的内部叫法是什么？", "长文档.pdf", ("蓝鲸计划",)),
    ("谈不拢的时候，双方走哪条路子解决分歧？", "合作协议.pdf", ("仲裁", "协商")),
]

MODES = [
    ("纯向量", "1.0", "0.0"),
    ("偏向量(0.9/0.1)", "0.9", "0.1"),
    ("混合(默认0.7/0.3)", "0.7", "0.3"),
    ("纯关键词", "0.0", "1.0"),
]


def prep() -> None:
    import eval_docs
    import main

    for name in ("合作协议.pdf", "产品手册.docx", "长文档.pdf", "合同扫描件.png"):
        try:
            main.delete_document(name, thread_id=THREAD)
        except Exception:
            pass
    main.add_document("合作协议.pdf", eval_docs.build_contract_pdf(), thread_id=THREAD)
    main.add_document("产品手册.docx", eval_docs.build_manual_docx(), thread_id=THREAD)
    main.add_document("长文档.pdf", eval_docs.build_long_pdf(), thread_id=THREAD)
    main.add_document("合同扫描件.png", eval_docs.build_scan_png(), thread_id=THREAD)


def main_() -> int:
    if os.path.isdir(os.environ["DATA_DIR"]):
        shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)

    import knowledge

    print("=" * 74)
    print("高难度检索对照：只描述意图、不复述原文词")
    print("=" * 74)
    prep()
    print(f"语料就绪，{len(CASES)} 条问题 × {len(MODES)} 种权重\n")

    table = {}
    for label, wv, wb in MODES:
        os.environ["DOC_VEC_WEIGHT"], os.environ["DOC_BM25_WEIGHT"] = wv, wb
        hit1 = hit3 = 0
        det = []
        t0 = time.perf_counter()
        for q, want_doc, words in CASES:
            hits = knowledge.search_documents(q, top_k=3, thread_id=THREAD)
            docs = [h["doc"] for h in hits]
            top1 = bool(docs) and docs[0] == want_doc
            top3 = want_doc in docs
            # 关键词也命中了才算「真的找到了那段」，避免同文档不同段落被算对
            kw = bool(hits) and any(w in hits[0]["text"] for w in words)
            hit1 += 1 if (top1 and kw) else 0
            hit3 += 1 if (top3 and any(any(w in h["text"] for w in words) for h in hits if h["doc"] == want_doc)) else 0
            det.append((q, want_doc, docs[:1], hits[0]["text"][:26] if hits else "", bool(top1 and kw)))
        dt = time.perf_counter() - t0
        table[label] = (hit1, hit3, dt / len(CASES))
        print(f"[{label}] top-1 {hit1}/{len(CASES)}  top-3 {hit3}/{len(CASES)}  平均 {dt / len(CASES) * 1000:.0f} ms")
        for q, want, got, snip, ok in det:
            if not ok:
                print(f"    ❌ {q[:22]}… 期望 {want}，top-1 是 {got}")

    print("\n" + "=" * 74)
    print(f"{'权重':<12}{'top-1':>10}{'top-3':>10}{'平均耗时':>12}")
    for label, (h1, h3, avg) in table.items():
        print(f"{label:<12}{h1}/{len(CASES):>7}{h3}/{len(CASES):>7}{avg * 1000:>10.0f} ms")

    base = table["混合(默认0.7/0.3)"]
    print("\n结论口径：混合 vs 纯关键词 = "
          f"top-1 {base[0]}/{len(CASES)} vs {table['纯关键词'][0]}/{len(CASES)}；"
          f"混合 vs 纯向量 = {base[0]}/{len(CASES)} vs {table['纯向量'][0]}/{len(CASES)}")

    # ---------- 第二段：同样的权重，谁把「可答/不可答」分得更开 ----------
    # 命中率只说明「找得到」，阈值好不好设取决于「可答与不可答的分数间隔」。
    unanswerable = [
        "今年的项目预算是多少",
        "甲方公司的统一社会信用代码",
        "张三的生日是什么时候",
        "上月销售额增长了多少",
    ]
    print("\n" + "=" * 74)
    print("分数间隔：可答题 top-1 分 vs 不可答题最高分（间隔越大，0.42 阈值越好设）")
    print("=" * 74)
    for label, wv, wb in MODES:
        os.environ["DOC_VEC_WEIGHT"], os.environ["DOC_BM25_WEIGHT"] = wv, wb
        ans_scores = []
        for q, _, _ in CASES:
            hits = knowledge.search_documents(q, top_k=1, thread_id=THREAD)
            if hits:
                ans_scores.append(hits[0]["score"])
        bad_scores = []
        for q in unanswerable:
            hits = knowledge.search_documents(q, top_k=1, thread_id=THREAD)
            if hits:
                bad_scores.append(hits[0]["score"])
        if ans_scores and bad_scores:
            gap = min(ans_scores) - max(bad_scores)
            print(f"  {label:<18} 可答最低 {min(ans_scores):.3f} / 不可答最高 {max(bad_scores):.3f}"
                  f"  →  间隔 {gap:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
