# -*- coding: utf-8 -*-
"""文档分析评测：自建评测集，量化检索命中率、拒答准确率、引用准确率。

- A 部分（检索指标）：只调用 embedding，评「正确答案所在片段是否进入 top-k」
- B 部分（回答指标）：调用模型，评「答案是否正确 / 无答案时是否正确拒答 / 引用是否指向对的文档」

运行：python eval_docs.py
"""
import io
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import main

THREAD = "eval-thread"


# ------------------------------------------------------------------ 语料

def build_contract_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 80), "第一章 总则", fontname="china-s", fontsize=18)
    p1.insert_text((72, 115), "本协议由甲方与乙方于 2024 年 3 月 15 日在武汉签订。",
                   fontname="china-s", fontsize=11)
    p1.insert_text((72, 135), "本协议适用于双方在软件开发、运维支持方面的全部合作事项。",
                   fontname="china-s", fontsize=11)
    p1.insert_text((72, 155), "任何一方违反本协议约定，应当承担相应的违约责任。",
                   fontname="china-s", fontsize=11)

    x0, y0, w, h = 72, 190, 180, 22
    for r in range(3):
        for c in range(2):
            p1.draw_rect(pymupdf.Rect(x0 + c * w, y0 + r * h, x0 + c * w + w, y0 + r * h + h),
                         color=(0, 0, 0), width=1)
    for r, row in enumerate([["项目", "金额"], ["违约金", "合同总额的 5%"], ["保证金", "10000 元"]]):
        for c, val in enumerate(row):
            p1.insert_text((x0 + c * w + 6, y0 + r * h + 15), val, fontname="china-s", fontsize=10)

    p2 = doc.new_page()
    p2.insert_text((72, 80), "第二章 争议解决", fontname="china-s", fontsize=18)
    p2.insert_text((72, 115), "双方发生争议应当先行友好协商；协商不成的，提交武汉仲裁委员会仲裁。",
                   fontname="china-s", fontsize=11)
    p2.insert_text((72, 150), "第三章 保密条款", fontname="china-s", fontsize=18)
    p2.insert_text((72, 185), "双方对合作中获知的商业秘密负有保密义务，保密期限为三年。",
                   fontname="china-s", fontsize=11)

    data = doc.tobytes()
    doc.close()
    return data


def build_manual_docx() -> bytes:
    from docx import Document

    d = Document()
    d.add_heading("产品概述", level=1)
    d.add_paragraph("本系统用于管理个人事务，覆盖备忘、待办、文档分析与日程提醒。")
    d.add_heading("性能指标", level=2)
    t = d.add_table(rows=3, cols=2)
    for r, row in enumerate([["指标", "目标"], ["单次回答响应时间", "< 2s"], ["文档检索命中率", "> 85%"]]):
        for c, val in enumerate(row):
            t.cell(r, c).text = val
    d.add_heading("数据安全", level=2)
    d.add_paragraph("所有对话数据保存在本地 SQLite 数据库，知识库按会话隔离，不上传第三方。")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def build_long_pdf(pages: int = 12, needle_page: int = 9) -> bytes:
    """长文档：多页填充内容，关键事实只出现在第 9 页，用于测试真实规模的检索能力。"""
    import pymupdf

    doc = pymupdf.open()
    for p in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text((72, 80), f"第{p}章 项目说明", fontname="china-s", fontsize=17)
        y = 115
        for i in range(1, 9):
            line = (
                f"本节说明第 {p} 页的第 {i} 项通用条款，涉及项目背景、组织分工与日常管理流程，"
                f"适用于所有参与方在本阶段的工作安排。"
            )
            page.insert_text((72, y), line, fontname="china-s", fontsize=10)
            y += 24
        if p == needle_page:
            page.insert_text((72, y + 10), "项目代号：蓝鲸计划，计划上线时间为 2026 年 9 月 30 日。",
                             fontname="china-s", fontsize=11)

    data = doc.tobytes()
    doc.close()
    return data


def build_scan_png() -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    from pathlib import Path

    img = Image.new("RGB", (1000, 240), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for cand in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(cand).exists():
            font = ImageFont.truetype(cand, 32)
            break
    if font is None:
        raise RuntimeError("找不到中文字体，无法生成 OCR 样例（容器内需安装 fonts-wqy-zenhei）")
    draw.text((30, 50), "合同编号：A-2024-001", fill="black", font=font)
    draw.text((30, 120), "签署日期：2024 年 3 月 15 日", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ 评测集
# (问题, 期望答案关键字, 期望文档关键字, 是否可答)
QA = [
    ("违约金比例是多少", "5%", "合作协议", True),
    ("保证金金额是多少", "10000", "合作协议", True),
    ("双方出现争议怎么解决", "仲裁", "合作协议", True),
    ("保密期限是几年", "三年", "合作协议", True),
    ("协议是什么时候签订的", "2024", "合作协议", True),
    ("合同编号是多少", "A-2024-001", "扫描件", True),
    ("产品的响应时间目标", "2s", "产品手册", True),
    ("文档检索命中率目标是多少", "85", "产品手册", True),
    ("系统的数据保存在哪里", "SQLite", "产品手册", True),
    ("这个系统主要用来做什么", "个人事务", "产品手册", True),
    # —— 长文档检索（关键事实只出现在第 9 页）——
    ("蓝鲸计划的上线时间是什么时候", ("2026 年 9 月 30 日", "2026-09-30", "9 月 30"), "长文档", True),
    ("项目代号是什么", "蓝鲸计划", "长文档", True),
    # —— 多步计算：比例在表格里，总额由问题给出 ——
    ("若合同总额为 20 万元，违约金应赔付多少钱", ("1 万", "10000", "一万元"), "合作协议", True),
    ("服务器 IP 地址是多少", None, None, False),
    ("公司一共有多少名员工", None, None, False),
    ("今年的项目预算是多少", None, None, False),
    ("甲方公司的统一社会信用代码", None, None, False),
]


def _kw_match(kw, text: str) -> bool:
    """关键字匹配，支持多个可接受答案。"""
    if kw is None:
        return False
    cands = kw if isinstance(kw, tuple) else (kw,)
    return any(c in text for c in cands)


def prep() -> None:
    # ask_document 内部靠 ContextVar 定位会话，这里显式绑定到评测会话
    main.bind_thread(THREAD)
    # 清理可能的残留
    for name in ("合作协议.pdf", "产品手册.docx", "合同扫描件.png", "长文档.pdf"):
        try:
            main.delete_document(name, thread_id=THREAD)
        except Exception:
            pass
    print(main.add_document("合作协议.pdf", build_contract_pdf(), thread_id=THREAD))
    print(main.add_document("产品手册.docx", build_manual_docx(), thread_id=THREAD))
    print(main.add_document("合同扫描件.png", build_scan_png(), thread_id=THREAD))
    print(main.add_document("长文档.pdf", build_long_pdf(), thread_id=THREAD))


def part_a() -> dict:
    """检索指标：top-1 / top-3 命中率。"""
    n = hit1 = hit3 = 0
    misses = []
    for q, kw, doc_kw, answerable in QA:
        if not answerable:
            continue
        n += 1
        hits = main.search_documents(q, top_k=3, thread_id=THREAD)
        texts = [h["text"] for h in hits]
        docs = [h["doc"] for h in hits]
        ok1 = bool(texts) and _kw_match(kw, texts[0]) and doc_kw in docs[0]
        ok3 = any(_kw_match(kw, t) and doc_kw in d for t, d in zip(texts, docs))
        hit1 += ok1
        hit3 += ok3
        if not ok3:
            misses.append((q, kw, texts[:1]))
    return {"n": n, "hit1": hit1, "hit3": hit3, "misses": misses}


def part_b() -> dict:
    """回答指标：正确率 / 拒答准确率 / 引用准确率。"""
    correct = refuse_ok = cite_ok = answered = 0
    details = []
    for q, kw, doc_kw, answerable in QA:
        raw = main.ask_document.invoke(q)
        if answerable:
            answered += 1
            ok = _kw_match(kw, raw)
            correct += ok
            cite = doc_kw in raw.split("依据：")[-1] if "依据：" in raw else False
            cite_ok += cite
            details.append(("可答", q, "✅" if ok else "❌", "引用✅" if cite else "引用❌"))
            if not ok:
                # 失败时把模型原答打出来，否则没法判断是检索错了还是算错了
                print(f"      ↳ 期望包含 {kw}；实际回答：{raw[:180].replace(chr(10), ' / ')}")
        else:
            refused = any(w in raw for w in ("没有", "未找到", "无相关", "无法", "不存在", "不清楚"))
            refuse_ok += refused
            details.append(("不可答", q, "✅拒答" if refused else "❌未拒答", ""))
    return {"answered": answered, "correct": correct,
            "unanswerable": len(QA) - answered, "refused": refuse_ok,
            "cited": cite_ok, "details": details}


# ------------------------------------------------------------------ 澄清追问评测
# (任务, 是否应该反问)
CLARIFY_CASES = [
    ("帮我设个提醒", True),
    ("提醒我一下", True),
    ("给我加个待办", True),
    ("现在几点了", False),
    ("明天十堰天气怎么样", False),
    ("看看我有哪些待办", False),
    ("生成今日日报", False),
    ("帮我制定一个考研复习计划", False),
]


def part_c() -> dict:
    """只跑规划节点，判断「该问时问、不该问时不问」。"""
    from langchain_core.messages import HumanMessage

    ok = 0
    details = []
    for task, should_ask in CLARIFY_CASES:
        state = {
            "messages": [HumanMessage(content=task)],
            "plan": "",
            "execution_result": "",
            "execution_trace": "",
            "review": "",
            "retry_count": 0,
        }
        out = main.planner_node(state)
        asked = str(out.get("plan", "")).startswith("需要补充")
        correct = asked == should_ask
        ok += correct
        details.append((task, should_ask, asked, correct))
    return {"n": len(CLARIFY_CASES), "ok": ok, "details": details}


def cleanup() -> None:
    for name in ("合作协议.pdf", "产品手册.docx", "合同扫描件.png", "长文档.pdf"):
        try:
            main.delete_document(name, thread_id=THREAD)
        except Exception:
            pass


# ------------------------------------------------------------------ 端到端（经状态图）
# A/B 段直接调工具函数，绕过 Planner/Executor；这一段走完整状态图，
# 专门防「文档明明上传了，规划节点却反问用户要文件」这类链路问题。
E2E_CASES = [
    ("合同里的违约金比例是多少", ("5%", "百分之五")),
    ("总结一下我上传的文档", ("违约", "合同")),
    ("这份合同的签订日期是哪天", ("2024", "3 月 15")),
]


def part_d() -> dict:
    from langchain_core.messages import HumanMessage

    ok = 0
    details = []
    graph = main.get_graph()
    thread = THREAD  # 必须与语料入库同一会话，否则规划节点看不到文档
    for q, kws in E2E_CASES:
        res = graph.invoke(
            {"messages": [HumanMessage(content=q)]},
            config={"configurable": {"thread_id": thread}},
        )
        ans = res.get("execution_result") or ""
        hit = any(k in ans for k in kws)
        used_doc = ("依据：" in ans) or ("《" in ans) or ("第" in ans and "页" in ans)
        good = hit and used_doc
        ok += good
        details.append((q, good, ans[:100]))
    return {"n": len(E2E_CASES), "ok": ok, "details": details}


def run() -> int:
    print("=== 准备语料 ===")
    prep()

    print("\n=== A. 检索指标 ===")
    a = part_a()
    r1 = a["hit1"] / a["n"] * 100
    r3 = a["hit3"] / a["n"] * 100
    print(f"  top-1 命中率：{a['hit1']}/{a['n']} = {r1:.1f}%")
    print(f"  top-3 命中率：{a['hit3']}/{a['n']} = {r3:.1f}%")
    for m in a["misses"]:
        print(f"    ❌ {m[0]}  期望含「{m[1]}」，实际 top1：{(m[2] or ['-'])[0][:40]}")

    print("\n=== B. 回答指标 ===")
    b = part_b()
    ac = b["correct"] / max(b["answered"], 1) * 100
    rc = b["refused"] / max(b["unanswerable"], 1) * 100
    cc = b["cited"] / max(b["answered"], 1) * 100
    for kind, q, mark, cite in b["details"]:
        print(f"  [{kind}] {q}  {mark} {cite}")
    print(f"\n  答案正确率：{b['correct']}/{b['answered']} = {ac:.1f}%")
    print(f"  拒答准确率：{b['refused']}/{b['unanswerable']} = {rc:.1f}%")
    print(f"  引用准确率：{b['cited']}/{b['answered']} = {cc:.1f}%")

    print("\n=== C. 澄清追问（只跑规划节点）===")
    c = part_c()
    cr = c["ok"] / c["n"] * 100
    for task, should_ask, asked, correct in c["details"]:
        mark = "✅" if correct else "❌"
        print(f"  {mark} {task:<22} 期望{'反问' if should_ask else '执行'}，实际{'反问' if asked else '执行'}")
    print(f"  澄清判定准确率：{c['ok']}/{c['n']} = {cr:.1f}%")

    print("\n=== D. 端到端（经完整状态图，验文档是否真的被用上）===")
    d = part_d()
    for q, good, ans in d["details"]:
        print(f"  {'✅' if good else '❌'} {q:<22} {(ans or '').replace(chr(10), ' ')[:80]}")
    dr = d["ok"] / d["n"] * 100
    print(f"  端到端正确率：{d['ok']}/{d['n']} = {dr:.1f}%")

    print("\n=== 汇总（可写入简历）===")
    print(f"  自建 {len(QA)} 条文档评测集：检索 top-3 命中率 {r3:.0f}%，"
          f"答案正确率 {ac:.0f}%，拒答准确率 {rc:.0f}%，引用准确率 {cc:.0f}%")
    print(f"  澄清判定：{c['ok']}/{c['n']} 条委托判断正确（{cr:.0f}%）")
    print(f"  端到端（状态图）：{d['ok']}/{d['n']}（{dr:.0f}%）")

    print("\n=== 清理 ===")
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(run())
