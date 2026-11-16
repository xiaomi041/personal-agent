# -*- coding: utf-8 -*-
"""澄清追问 + 读-析-写闭环 自检。

1) 信息不全的任务应触发反问，而不是瞎猜参数
2) 正常任务不应被误判为「需要补充」
3) 文档分析后可导出 docx（读 → 析 → 写）
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import eval_docs as E
import main

THREAD = "clarify-thread"


def ask(task: str) -> tuple[str, str]:
    """跑一轮图，返回 (最终答复, 计划)。"""
    g = main.get_graph()
    final, plan = "", ""
    for chunk in g.stream({"messages": [__import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage(content=task)]},
                          config={"configurable": {"thread_id": THREAD}},
                          stream_mode="updates"):
        for node, data in chunk.items():
            if node == "planner":
                plan = data.get("plan", plan)
                if data.get("execution_result"):
                    final = data["execution_result"]
            elif node == "executor":
                final = data.get("execution_result", final)
    return final, plan


def run() -> int:
    ok = True
    main._ACTIVE_THREAD = THREAD

    print("=== 1) 缺参数的委托应触发反问 ===")
    ans, plan = ask("帮我设个提醒")
    print("  答复：", ans[:120])
    if "需要补充" in plan or ans.startswith("🤔"):
        print("  ✅ 正确触发了澄清")
    else:
        print("  ❌ 没有触发澄清（可能直接瞎猜了参数）")
        ok = False

    print("\n=== 2) 正常委托不应误判 ===")
    ans2, plan2 = ask("现在几点了？")
    print("  答复：", ans2[:120])
    if plan2.startswith("需要补充"):
        print("  ❌ 误判为需要补充")
        ok = False
    else:
        print("  ✅ 正常执行")

    print("\n=== 3) 读 → 析 → 写 闭环 ===")
    main.add_document("合作协议.pdf", E.build_contract_pdf(), thread_id=THREAD)
    fields = main.extract_fields.invoke("违约金比例, 保证金, 保密期限")
    path = main.write_file.invoke({"filename": "合同要点.docx", "content": fields})
    print("  " + path)
    target = None
    for cand in (Path(path.split("：")[-1].strip()) if "：" in path else None,
                 main.DATA_DIR / "outputs" / "合同要点.docx",
                 Path.home() / "Desktop" / "合同要点.docx"):
        if cand and cand.exists():
            target = cand
            break
    if target:
        size = target.stat().st_size
        print(f"  ✅ 文件已生成：{target}（{size} 字节）")
        if size < 1000:
            print("  ⚠ 文件过小，内容可能为空")
            ok = False
    else:
        print("  ❌ 没找到生成的 docx")
        ok = False
    main.delete_document("合作协议.pdf", thread_id=THREAD)

    print("\n结果：" + ("通过 ✅" if ok else "存在问题 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
