# -*- coding: utf-8 -*-
"""产出文件：把文本/结论落成 md/docx/pptx，带路径回退与占用重试。"""
import logging
import os
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from settings import *  # noqa: F401,F403

log = logging.getLogger("agent.files")

def _is_heading_line(s: str) -> bool:
    """启发式判断一行是否是小标题：
    - 不以符号/数字开头
    - 不带冒号（带冒号的是"要点：说明"格式，属于内容）
    - 较短（≤20 字）且不以标点结尾
    """
    if not s:
        return False
    if s[0] in "-•*·0123456789":
        return False
    if ":" in s or "：" in s:
        return False
    if len(s) > 20:
        return False
    if s[-1] in "。：；，,.!！?？":
        return False
    return True

def _auto_pptx_pages(content: str) -> list:
    """LLM 没给 '---' 时，按小标题启发式自动分页。返回每页的行列表。"""
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if not lines:
        return []
    pages = []          # 每页 = [标题, 要点1, 要点2, ...]
    cur, items = None, []

    def flush():
        nonlocal cur, items
        if cur is not None:
            if items:
                pages.append([cur] + items)
            elif pages:
                pages[-1].append(cur)  # 孤立短句并入上一页
            cur, items = None, []

    # 封面：第一个标题行 + 紧随其后的非标题行作为副标题
    cover = [lines[0]]
    start = 1
    if len(lines) > 1 and not _is_heading_line(lines[1]):
        cover.append(lines[1].lstrip("-•* ").strip())
        start = 2

    for line in lines[start:]:
        if _is_heading_line(line):
            flush()
            cur = line
            items = []
        else:
            items.append(line.lstrip("-•* ").strip())
    flush()
    if not pages:  # 完全没识别出标题：按每 6 行强制分页
        body = lines[start:]
        for i in range(0, len(body), 6):
            pages.append([lines[0]] + body[i:i + 6])
    return [cover] + pages  # 第一页封面 + 后续内容页

def _output_dir() -> Path:
    """产出目录：本地优先桌面（打开就能看到），容器内用 DATA_DIR/outputs（挂载卷，重启不丢）。

    判断依据是「桌面是否本来就存在」——不能主动创建 ~/Desktop，
    否则容器里会凭空建一个 /root/Desktop，文件落在镜像层，容器一重建就没了。
    """
    desktop = Path(os.path.expanduser("~/Desktop"))
    candidates: list[Path] = []
    if desktop.is_dir():
        candidates.append(desktop)
    candidates.append(DATA_DIR / "outputs")

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return base
        except Exception:
            continue
    return DATA_DIR / "outputs"

def _save_with_fallback(out: Path, saver) -> Path:
    """保存文件；目标被占用（如在 WPS 中打开）时用 名称(1).后缀 重试。

    全部失败时必须抛错 —— 绝不能返回一个并不存在的路径，
    否则上层会显示「已生成文件」而磁盘上什么都没有。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        saver(out)
        return out
    except PermissionError as first:
        stem, ext = out.stem, out.suffix
        for i in range(1, 10):
            cand = out.with_name(f"{stem}({i}){ext}")
            try:
                saver(cand)
                return cand
            except PermissionError:
                continue
        raise PermissionError(f"无法写入 {out}（文件被占用或无写权限）") from first

@tool
def write_file(filename: str, content: str) -> str:
    """把内容生成文件保存，按扩展名自动识别格式。
    - .txt / .md：直接写文本
    - .docx：Word 文档，换行分段
    - .pptx：PPT。可用“---”分页，每页第一行是标题、其余行是该页要点；不写“---”也可以，代码会按小标题自动分页
    filename 必须带扩展名，如“周报.md”“计划.pptx”。
    本地保存到桌面；容器部署时（没有桌面目录）保存到 DATA_DIR/outputs。
    """
    from docx import Document
    from pptx import Presentation

    out = _output_dir() / Path(filename).name
    ext = out.suffix.lower()

    if ext in (".txt", ".md", ".csv", ".json", ".log"):
        out = _save_with_fallback(out, lambda p: p.write_text(content, encoding="utf-8"))
    elif ext == ".docx":
        doc = Document()
        doc.add_heading(out.stem, level=0)
        for line in content.split("\n"):
            line = line.strip()
            if line:
                doc.add_paragraph(line)
        out = _save_with_fallback(out, doc.save)
    elif ext == ".pptx":
        from pptx import Presentation as _Pres
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
        from pptx.enum.shapes import MSO_SHAPE

        DARK = RGBColor(0x1F, 0x3A, 0x5F)      # 主题深蓝
        ORANGE = RGBColor(0xE6, 0x7E, 0x22)    # 强调橙
        GRAY = RGBColor(0x33, 0x33, 0x33)      # 正文灰
        WHITE = RGBColor(0xFF, 0xFF, 0xFF)
        LIGHT = RGBColor(0xCF, 0xD8, 0xE6)

        prs = _Pres()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)

        # 双保险分页：显式 "---" 分段（≥2 段）优先；否则按小标题启发式自动分页
        raw = [p.strip() for p in content.split("---") if p.strip()]
        if len(raw) > 1:
            pages = [p.split("\n") for p in raw]
        else:
            pages = _auto_pptx_pages(content)

        for idx, page in enumerate(pages, 1):
            lines = [l.strip() for l in page if l.strip()]
            if not lines:
                continue
            slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白版式
            if idx == 1:
                # ===== 封面页：深蓝底 + 白色大标题居中 =====
                slide.background.fill.solid()
                slide.background.fill.fore_color.rgb = DARK
                tb = slide.shapes.add_textbox(Inches(1), Inches(2.6), Inches(11.33), Inches(1.6))
                tf = tb.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.alignment = PP_ALIGN.CENTER
                run = p.add_run()
                run.text = lines[0]
                run.font.size = Pt(44)
                run.font.bold = True
                run.font.color.rgb = WHITE
                if len(lines) > 1:
                    tb2 = slide.shapes.add_textbox(Inches(1.5), Inches(4.3), Inches(10.33), Inches(1.0))
                    tf2 = tb2.text_frame
                    tf2.word_wrap = True
                    p2 = tf2.paragraphs[0]
                    p2.alignment = PP_ALIGN.CENTER
                    run2 = p2.add_run()
                    run2.text = " ".join(lines[1:])
                    run2.font.size = Pt(18)
                    run2.font.color.rgb = LIGHT
            else:
                # ===== 内容页：深蓝标题条 + 项目符号正文 + 页码 =====
                bar = slide.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.1)
                )
                bar.fill.solid()
                bar.fill.fore_color.rgb = DARK
                bar.line.fill.background()
                tbx = slide.shapes.add_textbox(Inches(0.6), Inches(0.15), Inches(12), Inches(0.85))
                tf = tbx.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                run = p.add_run()
                run.text = lines[0]
                run.font.size = Pt(30)
                run.font.bold = True
                run.font.color.rgb = WHITE
                body = slide.shapes.add_textbox(Inches(0.9), Inches(1.6), Inches(11.5), Inches(5.0))
                tf = body.text_frame
                tf.word_wrap = True
                first = True
                for line in lines[1:]:
                    p = tf.paragraphs[0] if first else tf.add_paragraph()
                    first = False
                    clean = line.lstrip("-•* ").strip()  # 去掉内容自带的项目符号
                    run = p.add_run()
                    run.text = "• " + clean
                    run.font.size = Pt(22)
                    run.font.color.rgb = GRAY
                    p.space_after = Pt(14)
                pg = slide.shapes.add_textbox(Inches(12.2), Inches(7.05), Inches(0.9), Inches(0.4))
                p = pg.text_frame.paragraphs[0]
                p.alignment = PP_ALIGN.RIGHT
                run = p.add_run()
                run.text = str(idx)
                run.font.size = Pt(14)
                run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
        out = _save_with_fallback(out, prs.save)
    else:
        return f"不支持的文件类型：{ext}，仅支持 .txt/.md/.csv/.json/.log/.docx/.pptx"
    return f"已生成文件：{out}"

__all__ = [n for n in list(globals()) if not n.startswith("__")]
