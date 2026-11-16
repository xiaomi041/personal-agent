# -*- coding: utf-8 -*-
"""结构化抽取 / 多文档对比 自检。"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import eval_docs as E
import main

THREAD = "analysis-thread"


def run() -> int:
    main.bind_thread(THREAD)
    for name in ("合作协议.pdf", "产品手册.docx"):
        try:
            main.delete_document(name, thread_id=THREAD)
        except Exception:
            pass
    print(main.add_document("合作协议.pdf", E.build_contract_pdf(), thread_id=THREAD))
    print(main.add_document("产品手册.docx", E.build_manual_docx(), thread_id=THREAD))

    print("\n=== 结构化抽取 ===")
    out = main.extract_fields.invoke("甲方, 保证金, 违约金比例, 保密期限, 联系人")
    print(out)

    print("\n=== 多文档对比 ===")
    cmp_out = main.compare_documents.invoke(
        {"doc_a": "合作协议", "doc_b": "产品手册", "aspect": "数据与保密相关要求"}
    )
    print(cmp_out)

    for name in ("合作协议.pdf", "产品手册.docx"):
        try:
            main.delete_document(name, thread_id=THREAD)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(run())
