# -*- coding: utf-8 -*-
"""RAG 链路自检：入库 → 检索 → 阈值 → 全局摘要 → 清理。

需要 .env 里的 DEEPSEEK_API_KEY / QWEN_API_KEY（会真实调用 embedding 与模型，成本很低）。
运行：python selftest_rag.py
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import documents as D
import main
import selftest_documents as samples

THREAD = "selftest-thread"


def main_test() -> int:
    ok = True
    pdf = samples.make_pdf()

    print("=== 1) 入库 ===")
    msg = main.add_document("合同样例.pdf", pdf, thread_id=THREAD)
    print("  " + msg)

    print("\n=== 2) 幂等：重复上传同一文件 ===")
    msg2 = main.add_document("合同样例.pdf", pdf, thread_id=THREAD)
    print("  " + msg2)
    if "未变更" not in msg2:
        print("  ❌ 重复上传未跳过"); ok = False

    print("\n=== 3) 语义检索（命中）===")
    hits = main.search_documents("违约金比例是多少", thread_id=THREAD, top_k=3)
    for h in hits:
        cite = D.citation({**h, "doc": h["doc"]})
        print(f"  score={h['score']:.3f}  {cite}")
        print(f"       {h['text'][:70]}")
    if not hits:
        print("  ❌ 没检索到内容"); ok = False
    elif not any("违约金" in h["text"] or "5%" in h["text"] for h in hits):
        print("  ⚠ 命中片段未必包含答案")

    print("\n=== 4) 阈值：无关问题应返回空 ===")
    none_hits = main.search_documents("量子纠缠的贝尔不等式", thread_id=THREAD, top_k=3)
    print(f"  命中 {len(none_hits)} 条")
    if none_hits:
        print("  ⚠ 阈值可能偏低，无关问题仍召回了片段")
        for h in none_hits:
            print(f"     score={h['score']:.3f} {h['text'][:40]}")

    print("\n=== 5) 表格片段检索 ===")
    tb = main.search_documents("保证金多少钱", thread_id=THREAD, top_k=3)
    kinds = [h["kind"] for h in tb]
    print(f"  命中类型 {kinds}")
    if "table" not in kinds:
        print("  ⚠ 表格片段未命中")

    print("\n=== 6) 全局摘要（map-reduce）===")
    summary = main.summarize_document("总结这份合同的主要内容", THREAD)
    print("  " + summary.replace("\n", "\n  ")[:600])

    print("\n=== 7) 清理 ===")
    print("  " + main.delete_document("合同样例.pdf", thread_id=THREAD))

    print("\n结果：" + ("通过 ✅" if ok else "存在问题 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main_test())
