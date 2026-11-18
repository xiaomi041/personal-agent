"""验收「回答层引用校验」：宁可漏答，不许编。

判定只做两件事：
1. **可答的题**要给出答案，且答案带片段编号（能追溯到片段）；
2. **不可答的题**要么明说「未提及」，要么被降级——绝不允许给一个没依据的答案。

运行：python bench_cite.py
"""
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DATA_DIR", os.path.join(HERE, ".tmp-cite"))
sys.path.insert(0, HERE)

THREAD = f"cite-bench-{time.strftime('%m%d-%H%M%S')}"

ANSWERABLE = [
    "合同里的违约金比例是多少",
    "保证金金额是多少",
    "双方出现争议怎么解决",
    "保密期限是几年",
    "合同编号是多少",
    "产品的响应时间目标",
    "文档检索命中率目标是多少",
    "蓝鲸计划的上线时间是什么时候",
    "项目代号是什么",
    "签订日期是哪天",
]

UNANSWERABLE = [
    "今年的项目预算是多少",
    "甲方公司的统一社会信用代码",
    "张三的生日是什么时候",
    "上月销售额增长了多少",
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


def classify(text: str) -> str:
    """answered / refused / unsupported"""
    from knowledge import _cited_ids, _is_refusal

    if _is_refusal(text):
        return "refused"
    # 片段数量最多 8，编号超出就说明是模型自己编的
    if _cited_ids(text, 8):
        return "answered"
    return "unsupported"


def main_() -> int:
    if os.path.isdir(os.environ["DATA_DIR"]):
        shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)

    import runtime
    from knowledge import ask_document

    print("=" * 78)
    print("引用校验验收（宁可漏答，不许编）")
    print("=" * 78)
    prep()
    runtime.bind_thread(THREAD)
    fn = getattr(ask_document, "func", ask_document)

    bad = 0
    print("\n【可答题】应当答出且带编号")
    for q in ANSWERABLE:
        out = fn(q)
        kind = classify(out)
        if kind != "answered":
            bad += 1
            print(f"  ❌ {kind:<12} {q}")
            print(f"     ↳ {out[:110].replace(chr(10), ' ')}")
        else:
            print(f"  ✅ answered     {q}")

    print("\n【不可答题】应当拒答，不许编")
    for q in UNANSWERABLE:
        out = fn(q)
        kind = classify(out)
        if kind != "refused":
            bad += 1
            print(f"  ❌ {kind:<12} {q}")
            print(f"     ↳ {out[:110].replace(chr(10), ' ')}")
        else:
            print(f"  ✅ refused      {q}")

    snap = runtime.usage_snapshot()
    print("\n" + "=" * 78)
    print(f"引用校验计数：pass={snap['cite_pass']} repaired={snap['cite_repaired']} "
          f"rejected={snap['cite_rejected']}")
    print(f"模型调用：{snap['calls']} 次，输入 {snap['in']} / 输出 {snap['out']} token")
    print(f"不合格：{bad} 条（可答 {len(ANSWERABLE)} + 不可答 {len(UNANSWERABLE)}）")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main_())
