# -*- coding: utf-8 -*-
"""documents.py 自检脚本：生成样例文件 → 解析 → 打印结构化结果。

运行：python selftest_documents.py
"""
import io
import sys
import tempfile
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免中文/符号报 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import documents as D


def make_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "第一章 总则", fontname="china-s", fontsize=18)
    page.insert_text((72, 130), "本协议由甲乙双方于 2024 年 3 月签订，适用于全部合作事项。",
                     fontname="china-s", fontsize=11)
    page.insert_text((72, 150), "任何一方违约，应承担相应责任。", fontname="china-s", fontsize=11)

    # 画一个 3 行 2 列的表
    x0, y0, w, h = 72, 180, 200, 22
    for r in range(3):
        for c in range(2):
            rect = pymupdf.Rect(x0 + c * w, y0 + r * h, x0 + c * w + w, y0 + r * h + h)
            page.draw_rect(rect, color=(0, 0, 0), width=1)
    cells = [["项目", "金额"], ["违约金", "5%"], ["保证金", "10000 元"]]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            page.insert_text((x0 + c * w + 6, y0 + r * h + 15), val,
                             fontname="china-s", fontsize=10)

    page2 = doc.new_page()
    page2.insert_text((72, 90), "第二章 争议解决", fontname="china-s", fontsize=18)
    page2.insert_text((72, 130), "双方发生争议应先行协商，协商不成提交仲裁。",
                      fontname="china-s", fontsize=11)

    data = doc.tobytes()
    doc.close()
    return data


def make_docx() -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading("项目背景", level=1)
    doc.add_paragraph("本系统用于管理个人事务，覆盖备忘、待办与文档分析。")
    doc.add_heading("关键指标", level=2)
    table = doc.add_table(rows=3, cols=2)
    rows = [["指标", "目标"], ["响应时间", "< 2s"], ["检索命中率", "> 85%"]]
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            table.cell(r, c).text = val
    doc.add_paragraph("以上指标按月复盘。")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_image() -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for cand in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 容器内（Dockerfile 装了 fonts-wqy-zenhei）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(cand).exists():
            font = ImageFont.truetype(cand, 34)
            break
    if font is None:
        raise RuntimeError("找不到中文字体，无法生成 OCR 样例（Windows 用 msyh/simhei，容器用 fonts-wqy-zenhei）")
    draw.text((30, 40), "扫描件测试：合同编号 A-2024-001", fill="black", font=font)
    draw.text((30, 110), "签约日期 2024 年 3 月 15 日", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def show(title: str, blocks) -> None:
    print(f"\n=== {title} === 共 {len(blocks)} 块")
    for b in blocks:
        head = f"[{b.kind}|p{b.page}|{b.heading or '-'}]"
        text = b.text.replace("\n", " ⏎ ")
        print(f"  {head} {text[:90]}")


def main() -> int:
    ok = True

    pdf = make_pdf()
    pb = D.extract_blocks("合同样例.pdf", pdf)
    show("PDF 解析", pb)
    chunks = D.chunk_blocks(pb)
    pages = {c.page for c in chunks if c.page}
    kinds = {c.kind for c in chunks}
    has_table = "table" in kinds
    print(f"  分块 {len(chunks)} 个，页码 {sorted(pages)}，类型 {sorted(kinds)}")
    if len(pages) < 2:
        print("  ❌ 页码未覆盖两页"); ok = False
    if not has_table:
        print("  ❌ 未抽出表格"); ok = False
    if not any("第一章" in c.heading for c in chunks):
        print("  ⚠ 章节路径可能未识别")

    docx_bytes = make_docx()
    db = D.extract_blocks("报告样例.docx", docx_bytes)
    show("DOCX 解析", db)
    dchunks = D.chunk_blocks(db)
    if not any(c.kind == "table" for c in dchunks):
        print("  ❌ DOCX 表格未抽出"); ok = False
    if not any("关键指标" in c.heading for c in dchunks):
        print("  ❌ DOCX 章节路径丢失"); ok = False

    img = make_image()
    ib = D.extract_blocks("扫描件样例.png", img)
    show("图片 OCR", ib)
    if not ib or len(ib[0].text) < 4:
        print("  ❌ OCR 结果过短"); ok = False

    print("\n=== 引用格式 ===")
    for c in chunks[:3]:
        d = {"doc": "合同样例.pdf", "page": c.page, "heading": c.heading, "kind": c.kind}
        print("  " + D.citation(d))

    print("\n结果：" + ("全部通过 ✅" if ok else "存在问题 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
