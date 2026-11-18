"""为「自适应门控」标定参数：先量分布，再定规则。

背景：`DOC_MIN_SCORE=0.42` 是绝对阈值，在改写题上两头误判——
可答题最低 0.410（被拒），不可答题最高 0.639（被放过）。
想法是换成相对判据：**答案片段应该比同批候选明显高出一截**，
而不是「分数必须超过某个魔数」。

这个脚本只测量、不改产品逻辑：打印每个问题的
top1 / top-k 中位数 / 差距（gap），以及若干候选规则下的误判数。
跑完再决定要不要把规则写进 knowledge.search_documents。

运行：
    python bench_gate.py                 # 改写题（难）
    python bench_gate.py --easy          # eval_docs 里的 13 条（容易）
"""
import os
import shutil
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DATA_DIR", os.path.join(HERE, ".tmp-gate"))
os.environ["DOC_MIN_SCORE"] = "0.01"  # 本脚本自己判，不让产品阈值先过滤
sys.path.insert(0, HERE)

MODE = "easy" if "--easy" in sys.argv else "hard"
THREAD = f"gate-{MODE}-{time.strftime('%m%d-%H%M%S')}"

HARD = [
    ("如果我方交付延期，要赔给对方多少钱？", True),
    ("对方不守约定时，我们找哪个机构来评理？", True),
    ("合作中接触到的内部消息，我们得守口如瓶多久？", True),
    ("我们给对方留的那笔信用保证，金额是多少？", True),
    ("这份合作文件用什么编号归档？", True),
    ("这个系统回答一个问题，最多能让人等多久？", True),
    ("用户找资料时，多大比例要能一次就找对？", True),
    ("整个工程对外发布的日子定在哪天？", True),
    ("这个项目的内部叫法是什么？", True),
    ("谈不拢的时候，双方走哪条路子解决分歧？", True),
]
UNANSWERABLE = [
    ("今年的项目预算是多少", False),
    ("甲方公司的统一社会信用代码", False),
    ("张三的生日是什么时候", False),
    ("上月销售额增长了多少", False),
]


def prep() -> None:
    import eval_docs
    import main

    files = {
        "合作协议.pdf": eval_docs.build_contract_pdf,
        "产品手册.docx": eval_docs.build_manual_docx,
        "长文档.pdf": eval_docs.build_long_pdf,
        "合同扫描件.png": eval_docs.build_scan_png,
    }
    for name, build in files.items():
        try:
            main.delete_document(name, thread_id=THREAD)
        except Exception:
            pass
        main.add_document(name, build(), thread_id=THREAD)


def load_cases():
    if MODE == "easy":
        import eval_docs

        return [(q, True) for q, _kw, _doc, ok in eval_docs.QA if ok] + \
               [(q, False) for q, _kw, _doc, ok in eval_docs.QA if not ok]
    return HARD + UNANSWERABLE


def topk_scores(query: str, k: int = 5):
    import knowledge

    hits = knowledge.search_documents(query, top_k=k, thread_id=THREAD)
    return [h["score"] for h in hits]


def main_() -> int:
    if os.path.isdir(os.environ["DATA_DIR"]):
        shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)
    prep()
    cases = load_cases()

    print("=" * 80)
    print(f"门控标定（{MODE} 题集，{len(cases)} 条）：top1 / 中位数 / gap")
    print("=" * 80)
    rows = []
    for q, answerable in cases:
        scores = topk_scores(q)
        if not scores:
            rows.append((q, answerable, 0.0, 0.0, 0.0))
            continue
        top1 = scores[0]
        med = statistics.median(scores[1:]) if len(scores) > 1 else 0.0
        rows.append((q, answerable, top1, med, top1 - med))
        flag = "可答" if answerable else "不可答"
        print(f"  [{flag}] top1={top1:.3f} med={med:.3f} gap={top1 - med:+.3f}  {q[:26]}")

    ans = [r for r in rows if r[1]]
    un = [r for r in rows if not r[1]]
    if ans and un:
        print(f"\n  可答 top1 区间   {min(r[2] for r in ans):.3f} – {max(r[2] for r in ans):.3f}")
        print(f"  可答 gap  区间   {min(r[4] for r in ans):+.3f} – {max(r[4] for r in ans):+.3f}")
        print(f"  不可答 top1 区间 {min(r[2] for r in un):.3f} – {max(r[2] for r in un):.3f}")
        print(f"  不可答 gap  区间 {min(r[4] for r in un):+.3f} – {max(r[4] for r in un):+.3f}")

    # ---- 候选规则对比 ----
    print("\n" + "=" * 80)
    print("候选规则（「放行」= 判为有答案；可答题该放行，不可答题该拦下）")
    print("=" * 80)

    def rule_abs(score, gap, floor):
        return score >= floor

    def rule_gap(score, gap, floor, margin):
        return score >= floor and gap >= margin

    def rule_or(score, gap, floor, margin, hard):
        return score >= hard or (score >= floor and gap >= margin)

    rules = [
        ("绝对 0.42（现状）", lambda s, g: rule_abs(s, g, 0.42)),
        ("绝对 0.30", lambda s, g: rule_abs(s, g, 0.30)),
        ("gap≥0.05 且 ≥0.30", lambda s, g: rule_gap(s, g, 0.30, 0.05)),
        ("gap≥0.08 且 ≥0.30", lambda s, g: rule_gap(s, g, 0.30, 0.08)),
        ("gap≥0.10 且 ≥0.25", lambda s, g: rule_gap(s, g, 0.25, 0.10)),
        ("≥0.60 或 (gap≥0.08 且 ≥0.30)", lambda s, g: rule_or(s, g, 0.30, 0.08, 0.60)),
        ("≥0.55 或 (gap≥0.10 且 ≥0.28)", lambda s, g: rule_or(s, g, 0.28, 0.10, 0.55)),
    ]
    print(f"{'规则':<36}{'可答放行':>10}{'不可答误放':>12}{'合计错':>8}")
    for name, fn in rules:
        ok_ans = sum(1 for r in ans if fn(r[2], r[4]))
        bad_un = sum(1 for r in un if fn(r[2], r[4]))
        total = (len(ans) - ok_ans) + bad_un
        print(f"{name:<36}{ok_ans:>7}/{len(ans)}{bad_un:>9}/{len(un)}{total:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
