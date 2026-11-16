# -*- coding: utf-8 -*-
"""
文档解析与切片引擎（PersonalAI）
================================
把「上传文件 → 纯文本 → 定长切片」升级为「结构化解析 → 带元数据切片」：

- PDF：按页解析，保留页码；基于字号识别标题，维护章节路径；
        扫描页（抽不到文字）自动栅格化后走 OCR；表格用 pdfplumber 单独抽取。
- Word：按正文顺序遍历段落与表格（不再丢表格）；按 Heading 样式维护章节路径。
- Markdown：按 # 标题层级维护章节路径。
- 图片：直接 OCR。

切片规则：表格整块保留不切分；正文段落 ≤350 字保留，超长段落按 350 字 + 40 字重叠续切。
每个切片携带 {doc, idx, page, heading, kind}，供答案溯源到「第几页 · 哪一节」。
"""
from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 数据结构

@dataclass
class Block:
    """解析出的一个语义块（正文段落 / 表格 / 图片 OCR 结果）。"""
    text: str
    kind: str = "text"          # text | table | image
    page: int | None = None     # PDF 有页码；Word/Markdown 为 None
    heading: str = ""           # 章节路径，如 "第三章 违约责任 > 3.2 违约金"


@dataclass
class Chunk:
    """最终入库的切片。"""
    text: str
    page: int | None = None
    heading: str = ""
    kind: str = "text"


# ---------------------------------------------------------------- 工具函数

def sha256(content: bytes) -> str:
    """文件内容指纹，用于判断文档是否变更、避免重复向量化。"""
    return hashlib.sha256(content).hexdigest()


_HEADING_MAX_LEN = 60


def _looks_like_heading(line: str) -> bool:
    """启发式判断是否为标题行：短、无句末标点、像编号/章节。"""
    s = line.strip()
    if not s or len(s) > _HEADING_MAX_LEN:
        return False
    if s.endswith(("。", "，", "；", "：", ".", ",", ";", ":")):
        return False
    patterns = (
        r"^第[一二三四五六七八九十百零\d]+[章节部分篇]",
        r"^[一二三四五六七八九十]+[、.．]",
        r"^\d+(\.\d+)*[\s、.．]",
        r"^(附|附录|参考文献|目录|摘要|Abstract|结论|引言|前言)",
    )
    return any(re.match(p, s) for p in patterns)


def _join_lines(text: str) -> str:
    """把同一段落内被折行的文本接回去：中文直接相连，英文/数字之间补空格。"""
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    out = lines[0]
    for nxt in lines[1:]:
        prev_ch = out[-1] if out else ""
        next_ch = nxt[0] if nxt else ""
        # 两侧都是中文/中文标点 → 直接拼接；否则补空格
        cjk = lambda ch: "\u4e00" <= ch <= "\u9fff" or ch in "，。；：、（）《》“”‘’！？"
        out += nxt if (cjk(prev_ch) and cjk(next_ch)) else " " + nxt
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _most_common_size(spans: list[tuple[str, float]]) -> float:
    """取出现次数最多的字号作为正文字号（比中位数稳健，短页也不会误判）。"""
    from collections import Counter

    sizes = [round(s, 1) for _, s in spans if s > 0]
    if not sizes:
        return 0.0
    return Counter(sizes).most_common(1)[0][0]


# ---------------------------------------------------------------- OCR

_ocr_engine = None


def _ocr_local(img_bytes: bytes) -> str | None:
    """本地 RapidOCR（离线、免费）。未安装则返回 None。"""
    global _ocr_engine
    try:
        if _ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result, _ = _ocr_engine(np.array(img))
        if not result:
            return ""
        return "\n".join(item[1] for item in result)
    except ImportError:
        return None
    except Exception:
        return None


def _ocr_vl(img_bytes: bytes) -> str | None:
    """多模态模型兜底（Qwen-VL）。需要 QWEN_API_KEY 与外网。"""
    key = os.getenv("QWEN_API_KEY", "")
    if not key:
        return None
    import base64

    import requests

    b64 = base64.b64encode(img_bytes).decode()
    prompt = (
        "请提取这张图片中的全部文字内容，保持原有阅读顺序与结构；"
        "如果是表格，用 Markdown 表格输出；不要添加任何解释或总结。"
    )
    try:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": os.getenv("OCR_VL_MODEL", "qwen-vl-max"),
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "temperature": 0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def ocr_image(img_bytes: bytes) -> str:
    """图片 → 文本。优先本地 OCR，失败时回退多模态模型。

    OCR_ENGINE 环境变量可强制指定：local / vl / off
    """
    engine = os.getenv("OCR_ENGINE", "auto").lower()
    if engine == "off":
        raise ValueError("OCR 已关闭（OCR_ENGINE=off）")
    reasons: list[str] = []
    if engine in ("auto", "local"):
        text = _ocr_local(img_bytes)
        if text is not None:
            return text.strip()
        reasons.append("本地 OCR 不可用（rapidocr 未安装或缺少系统库，如 libxcb.so.1）")
    if engine in ("auto", "vl"):
        text = _ocr_vl(img_bytes)
        if text is not None:
            return text.strip()
        reasons.append("多模态 OCR 不可用（QWEN_API_KEY 未配置或接口调用失败）")
    raise ValueError("OCR 失败：" + "；".join(reasons))


# ---------------------------------------------------------------- 各格式解析

def _extract_markdown(content: bytes) -> list[Block]:
    """Markdown：按 # 标题维护章节路径。"""
    lines = content.decode("utf-8", errors="ignore").splitlines()
    blocks: list[Block] = []
    path: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append(Block(text=text, kind="text", heading=" > ".join(path)))
            buf.clear()

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            level = len(m.group(1))
            path = path[: level - 1]
            path.append(m.group(2).strip())
        else:
            buf.append(line)
    flush()
    return blocks


def _iter_docx_body(doc):
    """按文档正文顺序产出段落和表格（python-docx 默认只给段落）。"""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def _table_to_markdown(table) -> str:
    """docx 表格 → Markdown 表格文本（保住行列结构）。"""
    rows = []
    for row in table.rows:
        cells = [c.text.strip().replace("\n", " ") for c in row.cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, sep, *body])


def _extract_docx(content: bytes) -> list[Block]:
    """Word：按正文顺序解析，保留标题层级与表格。"""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(content))
    blocks: list[Block] = []
    path: list[str] = []

    for item in _iter_docx_body(doc):
        if isinstance(item, Table):
            md = _table_to_markdown(item)
            if md:
                blocks.append(Block(text=md, kind="table", heading=" > ".join(path)))
            continue

        assert isinstance(item, Paragraph)
        text = item.text.strip()
        if not text:
            continue
        style = (item.style.name or "") if item.style is not None else ""
        level = None
        m = re.match(r"^(?:Heading|标题)\s*(\d+)", style, re.I)
        if m:
            level = int(m.group(1))
        elif _looks_like_heading(text):
            level = len(path) + 1
        if level:
            path = path[: max(level - 1, 0)]
            path.append(text)
            continue
        blocks.append(Block(text=text, kind="text", heading=" > ".join(path)))
    return blocks


def _pdf_page_tables(page) -> list[str]:
    """用 pdfplumber 抽当前页表格，转成 Markdown。失败返回空列表。"""
    try:
        tables = page.find_tables()
        out = []
        for t in tables:
            rows = t.extract() or []
            rows = [[(c or "").strip().replace("\n", " ") for c in r] for r in rows if r]
            rows = [r for r in rows if any(r)]
            if len(rows) < 2:
                continue
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            md = "| " + " | ".join(rows[0]) + " |\n"
            md += "| " + " | ".join(["---"] * width) + " |\n"
            md += "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
            out.append(md)
        return out
    except Exception:
        return []


def _extract_pdf(content: bytes) -> list[Block]:
    """PDF：按页解析并保留页码；标题靠字号识别；扫描页自动 OCR；表格单独抽取。"""
    try:
        import pymupdf as fitz
    except ImportError:  # 兼容旧版
        import fitz  # type: ignore

    import pdfplumber

    blocks: list[Block] = []
    doc = fitz.open(stream=content, filetype="pdf")

    try:
        plumber = pdfplumber.open(io.BytesIO(content))
    except Exception:
        plumber = None

    # 预扫描：逐页收集文本行与字号，用「全文出现次数最多的字号」作为正文字号基线。
    # 只看单页会在「一页只有标题+一句话」时误判（标题字号出现次数反而最多）。
    page_spans: list[list[tuple[str, float]]] = []
    all_spans: list[tuple[str, float]] = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        spans: list[tuple[str, float]] = []
        for blk in page.get_text("dict").get("blocks", []):
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                if not text:
                    continue
                size = max((s.get("size", 0) for s in line.get("spans", [])), default=0)
                spans.append((text, size))
        page_spans.append(spans)
        all_spans.extend(spans)
    body_size = _most_common_size(all_spans)

    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            page_no = pno + 1
            headings: list[str] = []

            # 1) 先进表格：pdfplumber 抽表格，正文里重复出现的单元格文字稍后剔除
            table_cell_texts: set[str] = set()
            page_tables: list[str] = []
            if plumber is not None and pno < len(plumber.pages):
                page_tables = _pdf_page_tables(plumber.pages[pno])
                for md in page_tables:
                    for cell in md.split("|"):
                        cell = cell.strip()
                        if cell and cell != "---":
                            table_cell_texts.add(cell)

            # 2) 正文字块：用全文正文字号基线识别标题，短标题再补一层正则兜底
            spans = page_spans[pno]

            if spans:
                body_parts: list[str] = []
                for text, size in spans:
                    # 表格单元格文字优先剔除（否则会被正则误判成章节标题）
                    if text in table_cell_texts:
                        continue
                    is_heading = False
                    line_kind = "text"
                    if (
                        body_size > 0
                        and size >= body_size * 1.15
                        and len(text) <= _HEADING_MAX_LEN
                        and not text.endswith(("。", "，", "；"))
                    ):
                        is_heading = True
                        line_kind = "font"
                    elif (
                        body_size > 0
                        and size >= body_size * 0.95
                        and len(text) <= 20
                        and _looks_like_heading(text)
                    ):
                        # 字号接近正文但明显是章节标题（如「第二章 争议解决」）
                        is_heading = True
                        line_kind = "regex"
                    if is_heading:
                        headings.append(text)
                        if body_parts:
                            blocks.append(Block("\n".join(body_parts), "text", page_no, " > ".join(headings[:-1])))
                            body_parts = []
                        continue
                    # 表格里的单元格文字不再重复计入正文
                    if text in table_cell_texts:
                        continue
                    body_parts.append(text)
                if body_parts:
                    body_parts = [p for p in body_parts if p not in table_cell_texts]
                    if body_parts:
                        blocks.append(Block("\n".join(body_parts), "text", page_no, " > ".join(headings)))

            page_text = "\n".join(s for s, _ in spans).strip()

            # 3) 扫描页：抽不到文字 → 栅格化后 OCR
            if len(page_text) < 20:
                try:
                    pix = page.get_pixmap(dpi=200)
                    ocr_text = ocr_image(pix.tobytes("png")).strip()
                    if ocr_text:
                        blocks.append(Block(ocr_text, "image", page_no, " > ".join(headings)))
                except Exception:
                    pass

            # 3) 表格：pdfplumber 单独抽，避免错行
            if plumber is not None and pno < len(plumber.pages):
                for md in _pdf_page_tables(plumber.pages[pno]):
                    blocks.append(Block(md, "table", page_no, " > ".join(headings)))
    finally:
        doc.close()
        if plumber is not None:
            try:
                plumber.close()
            except Exception:
                pass
    return blocks


def _extract_plain(content: bytes) -> list[Block]:
    text = content.decode("utf-8", errors="ignore")
    return [Block(t.strip(), "text") for t in re.split(r"\n\s*\n", text) if t.strip()]


def _extract_image_file(content: bytes) -> list[Block]:
    text = ocr_image(content).strip()
    if not text:
        raise ValueError("图片中未识别到文字")
    return [Block(text, "image", 1)]


# ---------------------------------------------------------------- 对外入口

SUPPORTED = {".txt", ".md", ".markdown", ".docx", ".pdf",
             ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def extract_blocks(name: str, content: bytes) -> list[Block]:
    """按扩展名分发解析，返回结构化块列表。"""
    suffix = os.path.splitext(name)[1].lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"不支持的文件类型：{suffix}（支持 {' / '.join(sorted(SUPPORTED))}）")
    if suffix in (".txt",):
        return _extract_plain(content)
    if suffix in (".md", ".markdown"):
        return _extract_markdown(content)
    if suffix == ".docx":
        return _extract_docx(content)
    if suffix == ".pdf":
        return _extract_pdf(content)
    return _extract_image_file(content)


def chunk_blocks(blocks: list[Block], size: int = 350, overlap: int = 40) -> list[Chunk]:
    """结构化块 → 切片。表格整块保留；正文按段落聚合，超长段落带重叠续切。

    超长段落优先在句末标点处断开（窗口内回退找最后一个 。！？；），
    避免把一句话从中间切开，导致引用片段读起来断裂。
    """
    SENT_END = "。！？；!?;\n"

    def split_long(para: str) -> list[str]:
        out, start = [], 0
        while start < len(para):
            hard_end = min(start + size, len(para))
            if hard_end >= len(para):
                out.append(para[start:].strip())
                break
            # 在窗口后 60% 范围内回退到最近的句末标点
            window_start = start + int(size * 0.6)
            cut = -1
            for i in range(hard_end - 1, window_start - 1, -1):
                if para[i] in SENT_END:
                    cut = i + 1
                    break
            end = cut if cut > 0 else hard_end
            out.append(para[start:end].strip())
            start = max(end - overlap, start + 1)
        return [c for c in out if c]

    chunks: list[Chunk] = []
    for blk in blocks:
        text = blk.text.strip()
        if not text:
            continue
        if blk.kind == "table":
            chunks.append(Chunk(text, blk.page, blk.heading, "table"))
            continue
        for para in re.split(r"\n\s*\n", text):
            para = _join_lines(para)
            if not para:
                continue
            if len(para) <= size:
                chunks.append(Chunk(para, blk.page, blk.heading, blk.kind))
                continue
            for piece in split_long(para):
                chunks.append(Chunk(piece, blk.page, blk.heading, blk.kind))
    return chunks


def citation(chunk: dict) -> str:
    """生成可定位的引用描述，如「《合同.pdf》第 3 页 · 第三章 违约责任」。"""
    parts = [f"《{chunk.get('doc', '?')}》"]
    if chunk.get("page"):
        parts.append(f"第 {chunk['page']} 页")
    if chunk.get("heading"):
        parts.append(chunk["heading"].split(" > ")[-1])
    if chunk.get("kind") == "table":
        parts.append("表格")
    return " · ".join(parts)
