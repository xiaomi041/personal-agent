# -*- coding: utf-8 -*-
"""
个人委托智能体（Personal Delegation Agent）
========================================
基于 LangGraph 的可持久化个人委托智能体：
   用户委托任务 -> [规划节点] -> [执行节点] -> [评审节点] -> 返回结果
   - 规划节点：DeepSeek，把模糊任务拆成可执行计划
   - 执行节点：DeepSeek + Function Calling，真实调用工具完成任务
   - 评审节点：Qwen（异构模型），独立交叉校验执行结果
   - 持久化：SqliteSaver，对话与任务状态落盘，重启可续接

运行：python main.py
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict
from http.server import BaseHTTPRequestHandler

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

import os

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
QWEN_KEY = os.getenv("QWEN_API_KEY", "")

BASE_DIR = Path(__file__).parent

# 日志写到 agent.log（滚动，最多约 3MB）。原先后台线程和工具里的异常大多被静默吞掉，出问题无从查起
import logging
from logging.handlers import RotatingFileHandler

log = logging.getLogger("agent")
if not log.handlers:
    log.setLevel(logging.INFO)
    try:
        _fh = RotatingFileHandler(BASE_DIR / "agent.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(_fh)
    except OSError:
        log.addHandler(logging.NullHandler())
DB_FILE = BASE_DIR / "agent_memory.db"
# 备忘写到桌面，打开桌面就能看到
MEMO_FILE = Path(os.path.expanduser("~/Desktop")) / "memos.md"

if not DEEPSEEK_KEY or not QWEN_KEY:
    raise SystemExit("请在 .env 中配置 DEEPSEEK_API_KEY 和 QWEN_API_KEY")


# ========== 模型工厂（异构：DeepSeek + Qwen） ==========
def deepseek_model():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.3,
        timeout=60,      # 单次请求超时，避免网络抖动时整个委托卡死
        max_retries=2,   # 429/5xx/连接错误由 SDK 自动退避重试
    )


def qwen_model():
    return ChatOpenAI(
        model="qwen-plus",
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.2,
        timeout=60,
        max_retries=2,
    )


# ========== 工具（执行节点可调用） ==========
@tool
def get_today() -> str:
    """获取当前精确时间（年月日 时:分 星期）。用户说'今天''明天''两小时后''X小时后'等相对时间时必须先调用本工具。"""
    from datetime import date, datetime, timedelta

    now = datetime.now()
    wd = "一二三四五六日"[now.weekday()]
    return (
        f"现在 {now.strftime('%Y-%m-%d %H:%M:%S')} 星期{wd}；"
        f"明天 {(now.date() + timedelta(days=1)).isoformat()}；"
        f"后天 {(now.date() + timedelta(days=2)).isoformat()}"
    )


@tool
def write_memo(content: str) -> str:
    """把一条备忘/待办追加写入本地 memos.md 文件。"""
    with open(MEMO_FILE, "a", encoding="utf-8") as f:
        f.write(f"- {content}\n")
    return f"已写入备忘：{content}"


def list_memos() -> list:
    """逐条读取备忘（每行一条，含行首 '- '，按文件顺序）。"""
    try:
        if not MEMO_FILE.exists():
            return []
        return [l.rstrip() for l in MEMO_FILE.read_text(encoding="utf-8").splitlines()]
    except OSError:
        return []


def delete_memo(index: int) -> str:
    """按序号删除某条备忘（序号见 list_memos，从 1 开始）。"""
    try:
        lines = MEMO_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "读取备忘失败"
    if not 1 <= index <= len(lines):
        return f"序号无效，当前共 {len(lines)} 条备忘"
    removed = lines.pop(index - 1)
    MEMO_FILE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return f"已删除备忘：{removed.strip()}"


def clear_memos() -> str:
    """清空全部备忘。"""
    try:
        MEMO_FILE.write_text("", encoding="utf-8")
    except OSError:
        return "清空失败"
    return "已清空全部备忘"


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


def _unique_path(p: Path) -> Path:
    """目标已存在时自动加 (1)(2)... 后缀，避免静默覆盖已有文件；同时确保目录存在。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        return p
    for i in range(1, 100):
        cand = p.with_name(f"{p.stem}({i}){p.suffix}")
        if not cand.exists():
            return cand
    return p


def _save_with_fallback(out: Path, saver) -> Path:
    """保存文件；若目标被占用（如正在 WPS 中打开）则自动用 名称(1).后缀 重试。"""
    try:
        saver(out)
        return out
    except PermissionError:
        stem, ext = out.stem, out.suffix
        for i in range(1, 100):
            cand = out.with_name(f"{stem}({i}){ext}")
            try:
                saver(cand)
                return cand
            except PermissionError:
                continue
        return out


@tool
def write_file(filename: str, content: str) -> str:
    """把内容生成文件保存到桌面，按扩展名自动识别格式。
    - .txt / .md：直接写文本
    - .docx：Word 文档，换行分段
    - .pptx：PPT。可用“---”分页，每页第一行是标题、其余行是该页要点；不写“---”也可以，代码会按小标题自动分页
    filename 必须带扩展名，如“周报.md”“计划.pptx”。"""
    from docx import Document
    from pptx import Presentation

    # 只保留文件名部分：丢弃模型/网页可能带来的绝对路径或 ../，文件只会落在桌面
    safe_name = Path(str(filename).replace("\\", "/")).name
    if not safe_name:
        return "文件名无效"
    out = _unique_path(Path(os.path.expanduser("~/Desktop")) / safe_name)
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


# ---------- 待办管理 ----------
TODO_FILE = BASE_DIR / "todos.json"


def _load_todos() -> list:
    if TODO_FILE.exists():
        try:
            return json.loads(TODO_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_todos(todos: list) -> None:
    TODO_FILE.write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")


@tool
def add_todo(content: str) -> str:
    """添加一条待办事项（写入本地 todos.json）。"""
    todos = _load_todos()
    todos.append({
        "content": content,
        "done": False,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    _save_todos(todos)
    return f"已添加待办：{content}"


@tool
def list_todos() -> str:
    """列出全部待办，未完成在前并标序号，已完成打勾。"""
    todos = _load_todos()
    if not todos:
        return "当前没有待办事项"
    lines = []
    for i, t in enumerate(todos, 1):
        mark = "✅" if t.get("done") else "⬜"
        lines.append(f"{i}. {mark} {t['content']}（{t['created']}）")
    return "\n".join(lines)


@tool
def complete_todo(index: int) -> str:
    """按序号把某条待办标记为已完成（序号见 list_todos）。"""
    todos = _load_todos()
    if 1 <= index <= len(todos):
        todos[index - 1]["done"] = True
        _save_todos(todos)
        return f"已完成：{todos[index - 1]['content']}"
    return f"序号无效，当前共 {len(todos)} 条待办，请先 list_todos 查看"


def todo_items() -> list:
    """返回全部待办的结构化列表 [{content, done, created}]（供界面使用）。"""
    return _load_todos()


def delete_todo(index: int) -> str:
    """按序号删除某条待办（序号从 1 开始）。"""
    todos = _load_todos()
    if not 1 <= index <= len(todos):
        return f"序号无效，当前共 {len(todos)} 条待办"
    removed = todos.pop(index - 1)
    _save_todos(todos)
    return f"已删除待办：{removed.get('content', '')}"


def clear_todos() -> str:
    """清空全部待办。"""
    _save_todos([])
    return "已清空全部待办"


# ---------- 任务计划（拆解 + 进度跟踪） ----------
PLANS_FILE = BASE_DIR / "plans.json"


def _load_plans() -> list:
    if PLANS_FILE.exists():
        try:
            return json.loads(PLANS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_plans(plans: list) -> None:
    PLANS_FILE.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")


def _plan_breakdown(task: str, deadline: str = "") -> list:
    """调用 DeepSeek 把长期任务拆成 3-8 个可执行子任务（里程碑），失败时兜底拆 4 步。"""
    prompt = (
        "把下面这个长期任务拆解成 3-8 个可执行、可验收的子任务（里程碑），"
        "每个子任务一句话、按先后顺序排列。只输出 JSON 字符串数组，"
        '例如：["子任务1", "子任务2", ...]，不要输出其他任何内容。\n'
        f"任务：{task}\n" + (f"截止时间：{deadline}\n" if deadline else "")
    )
    try:
        resp = deepseek_model().invoke([HumanMessage(content=prompt)])
        text = (resp.content or "").strip()
        # 兼容模型输出外层 markdown 代码块的情况
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        arr = json.loads(text)
        if isinstance(arr, list) and arr:
            return [str(x).strip() for x in arr if str(x).strip()][:8]
    except Exception:
        pass
    return [f"{task} - 第{i}阶段" for i in range(1, 5)]


@tool
def create_plan(task: str, deadline: str = "") -> str:
    """把用户的长期目标/任务（如准备考试、学习计划、项目推进）自动拆解成子任务计划并保存。task: 任务描述；deadline: 截止日期（可选，格式 YYYY-MM-DD）。"""
    subs = _plan_breakdown(task, deadline)
    plans = _load_plans()
    pid = f"plan-{int(datetime.now().timestamp())}"
    plans.append({
        "id": pid,
        "task": task,
        "deadline": deadline,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtasks": [{"content": c, "status": "未开始"} for c in subs],
    })
    _save_plans(plans)
    lines = [f"已为「{task}」建立计划，共 {len(subs)} 步："]
    for i, c in enumerate(subs, 1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)


@tool
def list_plans() -> str:
    """列出全部任务计划及其进度（每步状态：未开始/进行中/已完成）。"""
    plans = _load_plans()
    if not plans:
        return "当前没有任务计划"
    lines = []
    for p in plans:
        subs = p.get("subtasks", [])
        done = sum(1 for s in subs if s.get("status") == "已完成")
        lines.append(f"计划「{p['task']}」({done}/{len(subs)} 已完成)"
                     + (f"，截止 {p['deadline']}" if p.get("deadline") else ""))
        for i, s in enumerate(subs, 1):
            lines.append(f"  {i}. [{s.get('status')}] {s['content']}")
    return "\n".join(lines)


@tool
def update_subtask(plan_id: str, index: int, status: str) -> str:
    """更新某个计划中子任务的状态。plan_id: 计划ID；index: 子任务序号（从1开始）；status: 未开始/进行中/已完成。"""
    plans = _load_plans()
    for p in plans:
        if p["id"] == plan_id:
            subs = p.get("subtasks", [])
            if 1 <= index <= len(subs):
                subs[index - 1]["status"] = status
                _save_plans(plans)
                return f"已更新「{p['task']}」第{index}步为「{status}」：{subs[index - 1]['content']}"
            return f"序号无效，该计划共 {len(subs)} 步"
    return f"未找到计划：{plan_id}"


@tool
def delete_plan(plan_id: str) -> str:
    """删除某个任务计划及其全部子任务。"""
    plans = _load_plans()
    for i, p in enumerate(plans):
        if p["id"] == plan_id:
            removed = plans.pop(i)
            _save_plans(plans)
            return f"已删除计划「{removed['task']}」"
    return f"未找到计划：{plan_id}"


def plan_items() -> list:
    """返回全部任务计划的结构化列表（供界面使用）。"""
    return _load_plans()


_WMO_CODE = {
    0: "晴", 1: "大部晴朗", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇", 51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
    56: "冻毛毛雨", 57: "冻毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "冻雨", 71: "小雪", 73: "中雪", 75: "大雪",
    77: "雪粒", 80: "阵雨", 81: "强阵雨", 82: "暴雨",
    85: "阵雪", 86: "强阵雪", 95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴",
}
_WIND_DIRS = ["北风", "东北风", "东风", "东南风", "南风", "西南风", "西风", "西北风"]


def _uv_level(uv: float) -> str:
    if uv < 3:
        return "弱"
    if uv < 6:
        return "中等"
    if uv < 8:
        return "强"
    if uv < 11:
        return "很强"
    return "极强"


@tool
def get_weather(city: str) -> str:
    """查询某城市未来7天天气预报：每天含天气现象、最低/最高温、紫外线指数、风向与风速（免费 Open-Meteo 接口）。"""
    import requests

    try:
        # 1) 城市名 -> 经纬度
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=10,
        ).json()
        results = geo.get("results") or []
        if not results:
            return f"未找到城市：{city}，请检查城市名"
        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        city_name = loc.get("name", city)

        # 2) 7 天逐日预报
        f = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,uv_index_max,wind_speed_10m_max,wind_direction_10m_dominant",
                "forecast_days": 7,
                "timezone": "Asia/Shanghai",
            },
            timeout=10,
        ).json()
        daily = f.get("daily", {})
        times = daily.get("time", [])
        if not times:
            return f"天气查询失败：接口未返回 {city_name} 的数据"

        lines = [f"{city_name}未来7天天气："]
        import datetime as _dt
        week = "一二三四五六日"
        for i, day in enumerate(times):
            code = daily["weather_code"][i]
            tmax = daily["temperature_2m_max"][i]
            tmin = daily["temperature_2m_min"][i]
            uv = daily["uv_index_max"][i]
            ws = daily["wind_speed_10m_max"][i]
            wd = daily["wind_direction_10m_dominant"][i]
            d = _dt.date.fromisoformat(day)
            label = ["今天", "明天", "后天"][i] if i < 3 else f"{d.month}-{d.day}"
            label += f" 周{week[d.weekday()]}"
            wind = _WIND_DIRS[int((wd + 22.5) // 45) % 8] if wd is not None else "—"
            lines.append(
                f"{label}（{day}）：{_WMO_CODE.get(code, '未知')}，"
                f"{tmin:.0f}~{tmax:.0f}℃，紫外线{uv:.0f}（{_uv_level(uv)}），"
                f"{wind}{ws:.0f}km/h"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"天气查询失败：{e}"


# ========== 文档问答库（RAG：解析 → 分块 → 向量化 → 检索） ==========
import re
import requests
import numpy as np

KB_DIR = BASE_DIR / "kb"
KB_CHUNKS = KB_DIR / "chunks.json"   # [{thread_id, doc, text}]
KB_VECS = KB_DIR / "vectors.npy"     # numpy 矩阵（行 = 片段）
KB_DOCS = KB_DIR / "docs.json"       # [{thread_id, name, chunk_count, created_at}]

# 当前执行会话（executor 节点设置，工具按此隔离知识库）
# 用 threading.local 而不是全局变量：多个浏览器标签同时运行时互不串会话
import threading

_ctx = threading.local()


def _active_thread() -> str:
    return getattr(_ctx, "thread_id", None) or "user-main"

EMBED_MODEL = "text-embedding-v3"
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


def _embed_texts(texts: list) -> list:
    """批量文本 → Qwen text-embedding-v3 向量（每次最多 10 条）。"""
    out = []
    for i in range(0, len(texts), 10):
        batch = texts[i : i + 10]
        for attempt in range(3):  # 向量化接口偶发超时/限流：最多重试 2 次（1s、2s 退避）
            try:
                resp = requests.post(
                    EMBED_URL,
                    headers={"Authorization": f"Bearer {QWEN_KEY}"},
                    json={"model": EMBED_MODEL, "input": batch},
                    timeout=60,
                )
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                _time.sleep(1 + attempt)
        data = resp.json()["data"]
        data.sort(key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)
    return out


def _extract_text(name: str, content: bytes) -> str:
    """按扩展名解析文件内容为纯文本。"""
    suffix = Path(name).suffix.lower()
    if suffix in (".txt", ".md"):
        return content.decode("utf-8", errors="ignore")
    if suffix == ".docx":
        from io import BytesIO

        from docx import Document

        doc = Document(BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if suffix == ".pdf":
        from io import BytesIO

        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"不支持的文件类型：{suffix}（支持 txt / md / docx / pdf）")


def _chunk_text(text: str, size: int = 350, overlap: int = 40) -> list:
    """按段落切块：每个段落作为完整语义单元；超长段落再按 size 字 + overlap 重叠续切。"""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            end = min(start + size, len(para))
            chunks.append(para[start:end].strip())
            if end >= len(para):
                break
            start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def _load_kb():
    chunks = (
        json.loads(KB_CHUNKS.read_text(encoding="utf-8"))
        if KB_CHUNKS.exists()
        else []
    )
    vecs = np.load(KB_VECS) if KB_VECS.exists() else np.zeros((0, 1024), dtype="float32")
    docs = (
        json.loads(KB_DOCS.read_text(encoding="utf-8")) if KB_DOCS.exists() else []
    )
    # 兼容旧数据：没有会话归属的记录归到 user-main
    for c in chunks:
        c.setdefault("thread_id", "user-main")
        c.setdefault("idx", 0)
    for d in docs:
        d.setdefault("thread_id", "user-main")
    return chunks, vecs, docs


def _save_kb(chunks, vecs, docs):
    """先写临时文件再 os.replace：中途失败不会留下半截文件，chunks 与 vectors 也不会错位。"""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    tmp_c = KB_CHUNKS.with_name(KB_CHUNKS.name + ".tmp")
    tmp_v = KB_VECS.with_name("vectors.tmp.npy")  # np.save 要求以 .npy 结尾
    tmp_d = KB_DOCS.with_name(KB_DOCS.name + ".tmp")
    tmp_c.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(tmp_v, vecs)
    tmp_d.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_v, KB_VECS)
    os.replace(tmp_c, KB_CHUNKS)
    os.replace(tmp_d, KB_DOCS)


_KB_LOCK = threading.RLock()


def _kb_locked(fn):
    """知识库的读-改-写必须串行，否则并发上传会互相覆盖。"""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with _KB_LOCK:
            return fn(*a, **kw)

    return wrapper


@_kb_locked
def add_document(filename: str, content: bytes, thread_id: str = "user-main") -> str:
    """解析一份上传的文档并入库（分块 → 向量化 → 追加存储），文档归属指定会话，仅该会话可见。"""
    text = _extract_text(filename, content)
    if len(text.strip()) < 20:
        raise ValueError("文档内容为空或无法提取文字")
    chunks = _chunk_text(text)
    vecs_new = np.array(_embed_texts(chunks), dtype="float32")
    chunks_old, vecs_old, docs = _load_kb()
    chunks_old.extend(
        {"thread_id": thread_id, "doc": filename, "idx": i, "text": c}
        for i, c in enumerate(chunks)
    )
    vecs = np.vstack([vecs_old, vecs_new]) if len(vecs_old) else vecs_new
    docs = [d for d in docs if not (d["thread_id"] == thread_id and d["name"] == filename)]
    docs.append(
        {
            "thread_id": thread_id,
            "name": filename,
            "chunk_count": len(chunks),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    _save_kb(chunks_old, vecs, docs)
    return f"已读入《{filename}》：{len(chunks)} 个知识片段，可以直接问我"


def list_documents(thread_id: str = None) -> list:
    """列出文档；thread_id 为空返回全部（管理用），否则只返回该会话的。"""
    _, _, docs = _load_kb()
    if thread_id is None:
        return docs
    return [d for d in docs if d["thread_id"] == thread_id]


@_kb_locked
def delete_document(filename: str, thread_id: str = None) -> str:
    """删除一个文档及其全部知识片段（重建索引）；thread_id 为空按文件名全局删除。"""
    chunks_old, vecs_old, docs = _load_kb()
    keep_idx = [
        i
        for i, c in enumerate(chunks_old)
        if not (c["doc"] == filename and (thread_id is None or c["thread_id"] == thread_id))
    ]
    chunks = [chunks_old[i] for i in keep_idx]
    dim = vecs_old.shape[1] if len(vecs_old) else 1024
    vecs = vecs_old[keep_idx] if keep_idx else np.zeros((0, dim), dtype="float32")
    docs = [
        d
        for d in docs
        if not (d["name"] == filename and (thread_id is None or d["thread_id"] == thread_id))
    ]
    _save_kb(chunks, vecs, docs)
    return f"已删除《{filename}》"


def search_documents(query: str, top_k: int = 5, thread_id: str = None) -> list:
    """语义检索：query 向量与库内所有片段做余弦相似度；thread_id 指定时只检索该会话文档。"""
    chunks, vecs, _ = _load_kb()
    if thread_id is not None:
        keep = [i for i, c in enumerate(chunks) if c["thread_id"] == thread_id]
        if not keep:
            return []
        chunks = [chunks[i] for i in keep]
        vecs = vecs[keep]
    if not chunks:
        return []
    qv = np.array(_embed_texts([query])[0], dtype="float32")
    qn = np.linalg.norm(qv) + 1e-9
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).ravel() + 1e-9
    scores = (vecs @ qv) / (norms * qn)
    idx = np.argsort(-scores)[:top_k]
    return [
        {
            "doc": chunks[i]["doc"],
            "idx": chunks[i].get("idx", 0),
            "text": chunks[i]["text"],
            "score": float(scores[i]),
        }
        for i in idx
    ]


@tool
def ask_document(query: str) -> str:
    """基于当前会话已上传的文档回答用户问题（RAG 检索 + 生成）。用户在问文档/资料/文件里写了什么时必须先调用本工具；本会话未上传文档时提示先上传。"""
    hits = search_documents(query, thread_id=_active_thread())
    if not hits:
        return "本会话还没有上传文档。请先在侧边栏「文档」区上传（支持 txt / md / docx / pdf）后再提问。"
    ctx = "\n\n".join(f"[片段{i+1}|《{h['doc']}》] {h['text']}" for i, h in enumerate(hits))
    prompt = (
        "你是文档问答助手。严格依据下面提供的文档片段回答用户问题；"
        "如果片段中没有相关信息，直接说明'知识库中没有相关内容'，不要编造。"
        "回答末尾用 [1][2] 标注引用的片段编号。\n\n"
        f"文档片段：\n{ctx}\n\n用户问题：{query}\n\n回答："
    )
    resp = deepseek_model().invoke([HumanMessage(content=prompt)])
    src = "、".join(dict.fromkeys(f"《{h['doc']}》片段#{h.get('idx', 0)}" for h in hits))
    return f"{resp.content}\n\n（依据：{src}）"


# ========== 定时提醒 / 自动任务（自研调度线程 + SQLite 持久化） ==========
import time as _time

_WEEKDAY_MAP = {
    "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
    "mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1, "wed": 2,
    "wednesday": 2, "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4, "sat": 5, "saturday": 5, "sun": 6, "sunday": 6,
}


def _db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            at TEXT NOT NULL,
            repeat TEXT NOT NULL DEFAULT 'once',
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            fired_at TEXT NOT NULL,
            shown INTEGER NOT NULL DEFAULT 0
        )"""
    )
    # 兼容旧库：补上动作型提醒需要的列（已存在则跳过）
    for col_sql in (
        "ALTER TABLE reminders ADD COLUMN action TEXT DEFAULT ''",
        "ALTER TABLE reminders ADD COLUMN action_arg TEXT DEFAULT ''",
        "ALTER TABLE reminders ADD COLUMN last_fired TEXT DEFAULT ''",
    ):
        try:
            conn.execute(col_sql)
        except sqlite3.OperationalError:
            pass
    return conn


def _parse_at(at: str, repeat: str):
    """把模型给的时间格式解析成 (datetime, 星期几, HH:MM)。"""
    if repeat == "once":
        # 同时支持 "YYYY-MM-DD HH:MM:SS"（秒级，用于“10秒后”）和 "YYYY-MM-DD HH:MM"
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(at.strip(), fmt), None, None
            except ValueError:
                continue
        raise ValueError(f"时间格式应为 YYYY-MM-DD HH:MM:SS 或 YYYY-MM-DD HH:MM，收到：{at}")
    if repeat == "daily":
        hm = at.strip()
        datetime.strptime(hm, "%H:%M")  # 校验格式
        return None, None, hm
    if repeat == "weekly":
        parts = at.strip().split()
        if len(parts) != 2:
            raise ValueError("weekly 格式应为：周X HH:MM")
        wd = re.sub(r"^(周|星期|礼拜|week)", "", parts[0].lower())  # 原 lstrip 会把 "wed" 剥成 "d"
        if wd in _WEEKDAY_MAP:
            wd = _WEEKDAY_MAP[wd]
        else:
            raise ValueError(f"无法识别星期：{parts[0]}")
        datetime.strptime(parts[1], "%H:%M")
        return None, wd, parts[1]
    raise ValueError(f"不支持的 repeat 类型：{repeat}")


@tool
def create_reminder(content: str, at: str, repeat: str = "once") -> str:
    """创建定时提醒/自动任务，到点后自动写入桌面备忘并推送到界面。
    repeat=once 时 at 为具体时间 "YYYY-MM-DD HH:MM:SS"（精确到秒，如 2026-10-13 08:00:30；用户说“X秒后/X分钟后”时必须先调 get_today 再换算到秒）；
    repeat=daily 时 at 为 "HH:MM"（每天这个时刻）；
    repeat=weekly 时 at 为 "周X HH:MM"（每周这一天这个时刻）。
    用户说"每天/每周/明早/后天 X 点提醒我 XX"时，先把相对时间换算成上述格式再调用。"""
    from datetime import timedelta

    _parse_at(at, repeat)
    conn = _db()
    conn.execute(
        "INSERT INTO reminders (content, at, repeat, created_at) VALUES (?,?,?,?)",
        (content, at, repeat, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()
    r = {"once": "一次性", "daily": "每天", "weekly": "每周"}[repeat]
    return f"已创建{r}提醒：{at} → {content}"


@tool
def list_reminders() -> str:
    """列出所有尚未执行的定时提醒。"""
    conn = _db()
    rows = conn.execute("SELECT id, content, at, repeat FROM reminders ORDER BY at").fetchall()
    conn.close()
    if not rows:
        return "当前没有未执行的定时提醒"
    return "\n".join(
        f"[{rid}] ({repeat} {at}) {content}" for rid, content, at, repeat in rows
    )


@tool
def cancel_reminder(reminder_id: int) -> str:
    """取消一个定时提醒（编号见 list_reminders）。"""
    conn = _db()
    cur = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()
    return f"已取消提醒 #{reminder_id}（{cur.rowcount} 条）" if cur.rowcount else f"提醒 #{reminder_id} 不存在"


# ========== 聚合日报任务链（每日报告） ==========
def _recent_memos(n: int = 10) -> list:
    """读取桌面 memos.md 最近 n 条备忘。"""
    try:
        if not MEMO_FILE.exists():
            return []
        lines = [
            l.strip().lstrip("- ").strip()
            for l in MEMO_FILE.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        return lines[-n:]
    except OSError:
        return []


def _recent_kb(n: int = 3) -> list:
    """知识库最近沉淀的片段（按保存顺序取末尾 n 条）。"""
    try:
        if not KB_CHUNKS.exists():
            return []
        data = json.loads(KB_CHUNKS.read_text(encoding="utf-8"))
        return [f"{c.get('doc', '文档')}：{c.get('text', '')[:60]}" for c in data[-n:]]
    except Exception:
        return []


@tool
def generate_daily_report(city: str = "十堰") -> str:
    """生成今日聚合日报：自动汇总 日期/城市天气/未完成待办/最近备忘/知识库最近沉淀，并由 AI 写一段今日总结，保存为桌面“日报_日期.md”。
    用户说“生成今日日报”“每日报告”“把今天的情况汇总一下”时调用；city 可指定城市（默认十堰）。"""
    from datetime import date as _date

    today = _date.today()
    wd = "一二三四五六日"[today.weekday()]
    weather = get_weather.invoke({"city": city}) if city else "（未指定城市，跳过天气）"
    todos = list_todos.invoke({})
    memos = _recent_memos()
    kb = _recent_kb()

    lines = [f"# 📅 每日报告 {today.isoformat()} 星期{wd}", ""]
    lines += [f"## ☀️ 天气（{city}）", weather, ""]
    lines += ["## ✅ 待办", todos, ""]
    lines += ["## 📝 最近备忘"]
    lines += [f"- {m}" for m in memos] if memos else ["（暂无）"]
    lines.append("")
    lines += ["## 📚 知识库最近沉淀"]
    lines += [f"- {k}" for k in kb] if kb else ["（暂无）"]
    lines.append("")
    body = "\n".join(lines)

    # AI 今日总结（失败则降级为无总结，不影响日报本身）
    try:
        resp = deepseek_model().invoke(
            [
                SystemMessage(
                    "你是个人助理，根据下面的天气、待办、备忘写 2-3 句简洁的今日总结，"
                    "给出行事建议，不要客套，不要复述全部内容。"
                ),
                HumanMessage(content=body),
            ]
        )
        body += "\n## 💡 今日总结\n" + str(resp.content) + "\n"
    except Exception:
        pass

    filename = f"日报_{today.isoformat()}.md"
    path = Path(os.path.expanduser("~/Desktop")) / filename
    try:
        path.write_text(body, encoding="utf-8")
        saved = str(path)
    except OSError:
        saved = f"桌面写入失败，内容如下：\n{body}"
    return f"已生成今日日报：{saved}\n\n{body}"


@tool
def schedule_daily_report(at: str, city: str = "十堰") -> str:
    """注册每日自动生成日报：每天 at（格式 HH:MM，如 08:00）时刻自动聚合 天气/待办/备忘/知识库 生成日报文件，并弹窗通知。
    用户说“每天早上 X 点生成日报”“定时日报”“每天自动汇总”时调用。"""
    _parse_at(at, "daily")  # 校验 HH:MM
    conn = _db()
    conn.execute(
        "INSERT INTO reminders (content, at, repeat, created_at, action, action_arg) "
        "VALUES (?,?,?,?,?,?)",
        (
            f"每日日报聚合（城市：{city}）",
            at,
            "daily",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "daily_report",
            city,
        ),
    )
    conn.commit()
    conn.close()
    return f"已注册每日日报任务：每天 {at} 自动生成（城市：{city}）"


def _is_due(at: str, repeat: str, now: datetime):
    try:
        dt, wd, hm = _parse_at(at, repeat)
    except Exception:
        return False
    if repeat == "once":
        return dt <= now
    if repeat == "daily":
        return now.strftime("%H:%M") == hm
    if repeat == "weekly":
        return now.weekday() == wd and now.strftime("%H:%M") == hm
    return False


def _advance_at(at: str, repeat: str):
    """触发后推进下次时间。"""
    from datetime import timedelta

    if repeat == "daily":
        return (datetime.now() + timedelta(days=1)).strftime("%H:%M")
    if repeat == "weekly":
        _, wd, hm = _parse_at(at, repeat)
        return f"{'一二三四五六日'[wd]} {hm}"
    return at


def _check_due():
    from datetime import timedelta

    now = datetime.now()
    conn = _db()
    rows = conn.execute(
        "SELECT id, content, at, repeat, action, action_arg, last_fired FROM reminders"
    ).fetchall()
    for rid, content, at, repeat, action, action_arg, last_fired in rows:
        if not _is_due(at, repeat, now):
            continue
        fired_at = now.strftime("%Y-%m-%d %H:%M")
        # 去重：daily/weekly 按“分钟”匹配，轮询间隔内会命中多次，同一分钟只触发一次
        if repeat != "once" and (last_fired or "") == fired_at:
            continue
        alert_text = f"⏰ {content}"
        if action == "daily_report":
            conn.commit()  # 先提交，避免联网生成日报期间一直占着数据库写锁
            # 动作型提醒：到点真正执行“聚合日报”任务链
            try:
                report = generate_daily_report(city=action_arg or "")
                head, _, rest = report.partition("\n")
                preview = "\n".join(rest.split("\n")[:3])
                alert_text = f"📅 今日日报已生成\n{head}\n{preview}..."
            except Exception as e:
                log.exception("定时日报生成失败")
                alert_text = f"📅 日报生成失败：{e}"
        conn.execute(
            "INSERT INTO alerts (content, fired_at) VALUES (?,?)",
            (alert_text, fired_at),
        )
        try:
            with open(MEMO_FILE, "a", encoding="utf-8") as f:
                f.write(f"- [提醒 {fired_at}] {alert_text}\n")
        except OSError:
            pass
        if repeat == "once":
            conn.execute("DELETE FROM reminders WHERE id = ?", (rid,))
        else:
            conn.execute(
                "UPDATE reminders SET last_fired=? WHERE id=?",
                (fired_at, rid),
            )
    conn.commit()
    conn.close()


_scheduler_started = False


def start_scheduler():
    """启动后台调度线程（幂等）：每 5 秒检查一次到期提醒。"""
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True

    def _loop():
        while True:
            try:
                _check_due()
            except Exception:
                log.exception("调度循环出错")
            _time.sleep(5)

    import threading

    t = threading.Thread(target=_loop, daemon=True)
    t.start()

    # 顺带启动本地提醒 HTTP 服务（前端 JS 轮询用，与 Streamlit 页面刷新完全解耦）
    start_alert_http_server()


# ========== 本地提醒 HTTP 服务（供前端 JS 轮询弹窗，避免 Streamlit 定时刷新冲突） ==========
_ALERT_HTTP_PORT = 8600
_alert_server = None


class _AlertHandler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse

        try:
            u = urlparse(self.path)
            if u.path == "/alerts":
                rows = get_unread_alerts()
                self._send(200, rows)
            else:
                self._send(404, {"error": "not found"})
        except Exception:
            self._send(500, {"error": "internal"})

    def do_POST(self):
        """确认提醒必须用 POST：GET 不应改变状态，也避免被网页跨站请求误触发。"""
        from urllib.parse import parse_qs, urlparse

        try:
            u = urlparse(self.path)
            if u.path == "/ack":
                qs = parse_qs(u.query)
                aid = qs.get("id", [""])[0]
                if aid:
                    conn = _db()
                    conn.execute("UPDATE alerts SET shown=1 WHERE id=?", (aid,))
                    conn.commit()
                    conn.close()
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except Exception:
            self._send(500, {"error": "internal"})

    def log_message(self, *args):
        pass


def start_alert_http_server():
    global _alert_server
    if _alert_server:
        return
    import threading
    from http.server import ThreadingHTTPServer

    try:
        _alert_server = ThreadingHTTPServer(("127.0.0.1", _ALERT_HTTP_PORT), _AlertHandler)
        threading.Thread(target=_alert_server.serve_forever, daemon=True).start()
    except OSError as e:
        # 端口被占用说明服务已在运行；否则写日志便于排查
        try:
            Path(__file__).parent.joinpath("_alert_http.log").write_text(
                f"bind failed: {e!r}", encoding="utf-8"
            )
        except Exception:
            pass


def get_unread_alerts() -> list:
    conn = _db()
    rows = conn.execute(
        "SELECT id, content, fired_at FROM alerts WHERE shown=0 ORDER BY id"
    ).fetchall()
    conn.close()
    return [{"id": i, "content": c, "fired_at": t} for i, c, t in rows]


def mark_alerts_shown():
    conn = _db()
    conn.execute("UPDATE alerts SET shown=1 WHERE shown=0")
    conn.commit()
    conn.close()


# ========== 联网搜索 → 知识库（Bing 免 key + 复用 RAG 入库） ==========
_SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def _bing_search(query: str, num: int = 5) -> list:
    """Bing 网页搜索（免 key，国内可访问），返回标题/链接/摘要。"""
    r = requests.get(
        "https://cn.bing.com/search",
        params={"q": query, "mkt": "zh-CN"},
        headers=_SEARCH_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    items = re.findall(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text)
    snips = re.findall(r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', r.text, re.S)
    snips = [re.sub(r"<[^>]+>", "", s).strip() for s in snips]
    out = []
    for i in range(min(num, len(items))):
        title = re.sub(r"<[^>]+>", "", items[i][1]).strip()
        out.append(
            {
                "title": title,
                "link": items[i][0],
                "snippet": snips[i] if i < len(snips) else "",
            }
        )
    return out


def _compile_note(query: str, body: str) -> str:
    """把搜索结果整理成 Markdown 笔记。"""
    prompt = (
        "把下面的搜索结果整理成一份结构清晰的 Markdown 笔记："
        "标题、2-4 个要点小节、每条信息附来源链接，只基于给定内容，不编造。\n\n" + body
    )
    resp = deepseek_model().invoke([HumanMessage(content=prompt)])
    return f"# {query}\n\n{resp.content}"


@tool
def web_search(query: str, save_to_kb: bool = True) -> str:
    """联网搜索（Bing 免 key）。save_to_kb=True（默认）时把结果整理成 Markdown 笔记自动存入知识库，下次可直接基于它问答；False 时只返回搜索结果。"""
    try:
        results = _bing_search(query)
    except Exception as e:
        return f"搜索失败：{type(e).__name__}: {e}"
    if not results:
        return f"未搜索到「{query}」的相关结果，请换关键词再试"
    lines = [
        f"{i+1}. {r['title']}\n   链接：{r['link']}\n   摘要：{r['snippet']}"
        for i, r in enumerate(results)
    ]
    body = "\n\n".join(lines)
    if not save_to_kb:
        return f"「{query}」搜索结果：\n\n{body}"
    try:
        note = _compile_note(query, body)
        safe = re.sub(r'[\\/:*?"<>|]', "", query)[:12].strip() or "搜索"
        filename = f"搜索笔记-{safe}.md"
        add_document(filename, note.encode("utf-8"), thread_id=_active_thread())
        return f"「{query}」搜索结果：\n\n{body}\n\n📚 已整理成《{filename}》存入知识库，下次可直接问我"
    except Exception as e:
        return f"「{query}」搜索结果：\n\n{body}\n\n（笔记入库失败：{e}）"


TOOLS = [
    get_today, write_memo, write_file, add_todo, list_todos, complete_todo,
    get_weather, ask_document, create_reminder, list_reminders,
    cancel_reminder, web_search, generate_daily_report, schedule_daily_report,
    create_plan, list_plans, update_subtask, delete_plan,
]
TOOL_MAP = {t.name: t for t in TOOLS}


# ========== 状态定义 ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 全部对话历史（持久化）
    plan: str
    execution_result: str
    execution_trace: str  # 工具调用证据链（供评审核验）
    review: str
    retry_count: int  # 评审返工次数（限制最多重做 2 轮）


def _last_user_task(state) -> str:
    """取当前这条用户委托（最后一条 HumanMessage），不能用 messages[0]（多轮后第一条是历史）。"""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def _build_context(state) -> str:
    """把整个会话的对话（用户问 + 助手答）整理成文本，让 planner/executor 能理解多轮指代。
    跳过 [规划]/[评审] 等内部节点消息和工具调用噪音，只保留问答主干。"""
    pairs = []          # [(用户问题, 助手最终答复 or None), ...]
    user, ans = None, []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            if user is not None:
                pairs.append((user, ans[-1] if ans else None))
            user = m.content
            ans = []
        elif isinstance(m, AIMessage):
            c = (m.content or "").strip()
            if not c or c.startswith("[规划]") or c.startswith("[评审]"):
                continue
            if user is not None:
                ans = [c]  # 保留该轮最后一条真实答复
    if user is not None:
        pairs.append((user, ans[-1] if ans else None))
    lines = []
    for u, a in pairs:
        lines.append(f"用户：{u}")
        if a:
            lines.append(f"助手：{a}")
    return "\n".join(lines)


# ========== 节点1：规划（DeepSeek） ==========
PLANNER_PROMPT = SystemMessage(
    "你是任务规划师。把用户委托的任务拆成 1-3 步可执行计划，"
    "只输出计划本身，格式：'计划：①... ②... ③...'，不要执行任务，不要解释。"
)


def planner_node(state: State):
    user_task = _last_user_task(state)
    context = _build_context(state)
    prompt = f"对话历史：\n{context}\n\n当前任务：{user_task}"
    resp = deepseek_model().invoke([PLANNER_PROMPT, HumanMessage(content=prompt)])
    plan = resp.content
    # 每个新委托从 planner 进入：重置上一轮残留的评审状态，否则 retry_count 会跨轮累加
    # （返工机制只在会话第 1 条消息有效），上一轮的评审意见也会混进本轮 executor 的提示
    return {
        "plan": plan,
        "review": "",
        "retry_count": 0,
        "execution_result": "",
        "execution_trace": "",
        "messages": [AIMessage(content=f"[规划] {plan}")],
    }


# ========== 节点2：执行（DeepSeek + 工具循环） ==========
EXECUTOR_PROMPT = SystemMessage(
    "你是执行专员。严格按照计划调用可用工具完成任务："
    "1. 需要日期时先调 get_today；"
    "2. 需要记录备忘时调 write_memo；"
    "3. 用户询问已上传文档/知识库里的内容时，必须调 ask_document；"
    "4. 用户要求生成今日/每日聚合日报、汇总报告时，调 generate_daily_report；"
    "5. 用户要求每天早上定时生成日报（“每天早上 X 点”）时，调 schedule_daily_report；"
    "6. 用户委托长期目标（准备考试/学习计划/项目推进/养成习惯等）要求拆解规划时，调 create_plan 自动拆解并保存；"
    "7. 用户汇报某步进展（开始做/做完了）时，先调 list_plans 找到对应计划，再调 update_subtask 更新状态；"
    "8. 工具调用完成后，用一句话总结执行结果。"
)


def _run_tool_loop(model, messages, max_iters=6):
    """Function Calling 循环：模型请求工具 -> 执行 -> 把结果回喂，直到模型不再请求工具。
    返回 (最终回答, 全部消息, 工具调用证据链)。"""
    last = None
    trace = []
    for _ in range(max_iters):
        last = model.invoke(messages)
        messages = messages + [last]
        if not getattr(last, "tool_calls", None):
            return last, messages, trace
        for tc in last.tool_calls:
            fn = TOOL_MAP.get(tc["name"])
            if fn is None:
                result = f"未知工具：{tc['name']}"
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            else:
                try:
                    result = fn.invoke(tc["args"])
                except Exception as e:
                    log.exception("工具 %s 执行出错", tc["name"])
                    result = f"工具执行出错：{e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            trace.append(f"调用 {tc['name']}({tc['args']}) -> {result}")
    return last, messages, trace


def executor_node(state: State, config=None):
    if config and config.get("configurable", {}).get("thread_id"):
        _ctx.thread_id = config["configurable"]["thread_id"]
    user_task = _last_user_task(state)  # 当前这条委托，而不是第一条历史
    review_note = (
        f"\n评审意见（若指出问题，必须按意见修正后重新执行）：\n{state['review']}"
        if state.get("review")
        else ""
    )
    context = _build_context(state)
    seed = [
        EXECUTOR_PROMPT,
        HumanMessage(
            content=f"对话历史：\n{context}\n\n用户任务：{user_task}\n执行计划：{state['plan']}{review_note}"
        ),
    ]
    model = deepseek_model().bind_tools(TOOLS)
    final, messages, trace = _run_tool_loop(model, seed)
    shown = [m for m in messages if isinstance(m, (AIMessage, ToolMessage))]
    return {
        "execution_result": final.content,
        "execution_trace": "\n".join(trace) if trace else "（本轮未调用工具，纯文本回答）",
        "messages": shown,
    }


# ========== 节点3：评审（Qwen 异构模型交叉校验） ==========
CRITIC_PROMPT = SystemMessage(
    "你是严格的委托任务评审专家，独立检查执行结果：\n"
    "1. 计划是否被完整执行；\n"
    "2. 根据提供的工具调用证据链判断执行是否真实发生（不要臆测工具没调用）；\n"
    "3. 结果是否满足用户原始需求；\n"
    "4. 是否有遗漏或明显错误。\n"
    "注意：你看到的具体日期来自真实工具返回值，以证据链为准，不要凭自己的知识否定。\n"
    "输出格式：先给结论'通过'或'需修正'，再列具体问题（每条一行），不要复述执行细节。"
)


def critic_node(state: State):
    user_task = _last_user_task(state)  # 当前这条委托，而不是第一条历史
    context = _build_context(state)
    material = (
        f"对话历史：\n{context}\n\n"
        f"用户委托：{user_task}\n"
        f"执行计划：{state['plan']}\n"
        f"工具调用证据链：\n{state['execution_trace']}\n"
        f"执行结果：{state['execution_result']}"
    )
    resp = qwen_model().invoke([CRITIC_PROMPT, HumanMessage(content=material)])
    return {
        "review": resp.content,
        "messages": [AIMessage(content=f"[评审] {resp.content}")],
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ========== 条件路由：评审"需修正"→ 回执行重做（最多 MAX_RETRY 轮） ==========
MAX_RETRY = 2  # 最多返工次数（critic 共最多执行 MAX_RETRY + 1 次）


def _needs_fix(review: str) -> bool:
    """评审结论是否为“需修正”：只看开头结论，避免“无需修正/不需修正”被子串误判。"""
    head = (review or "").strip()[:12]
    return "需修正" in head and "无需修正" not in head and "不需修正" not in head


def _review_router(state: State) -> str:
    """评审说"需修正"且未超重做上限 → 回 executor 返工；否则结束。"""
    if _needs_fix(state.get("review", "")) and state.get("retry_count", 0) <= MAX_RETRY:
        return "executor"
    return "end"


# ========== 构建状态图 ==========
def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "critic")
    # 评审"需修正"→ 回 executor 重做（自主决策循环）；通过 → 结束
    g.add_conditional_edges("critic", _review_router, {"executor": "executor", "end": END})
    return g.compile(checkpointer=checkpointer)


def get_graph():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_graph(checkpointer)


# ========== 会话管理（回访历史会话） ==========
SESSIONS_FILE = BASE_DIR / "sessions.json"


def _load_sessions() -> list:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_sessions(recs: list) -> None:
    SESSIONS_FILE.write_text(
        json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def register_session(thread_id: str, title: str) -> None:
    """记录一个会话。新会话默认标题「新会话」；输入首条内容后才按内容更新标题。"""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recs = _load_sessions()
    for r in recs:
        if r["thread_id"] == thread_id:
            # 已有实质标题的不覆盖；空/占位标题在有输入时更新为内容
            if title and (not r.get("title") or r["title"] in ("新会话",) or r["title"].startswith("会话 ")):
                r["title"] = title[:24]
            r["created_at"] = now
            _save_sessions(recs)
            return
    recs.append(
        {
            "thread_id": thread_id,
            "created_at": now,
            "title": title[:24] if title else "新会话",
            "pinned": False,
        }
    )
    _save_sessions(recs)


def set_pinned(thread_id: str, pinned: bool) -> None:
    """置顶 / 取消置顶一个会话。"""
    recs = _load_sessions()
    for r in recs:
        if r["thread_id"] == thread_id:
            r["pinned"] = pinned
            _save_sessions(recs)
            return
    from datetime import datetime

    recs.append(
        {
            "thread_id": thread_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": "新会话",
            "pinned": pinned,
        }
    )
    _save_sessions(recs)


def list_sessions() -> list:
    """所有历史会话（sessions.json 优先；再扫描数据库补漏），新的在前。"""
    recs = _load_sessions()
    known = {r["thread_id"] for r in recs}
    try:
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        conn.close()
        for (tid,) in rows:
            tid = tid.decode() if isinstance(tid, bytes) else str(tid)
            if tid not in known:
                recs.append(
                    {"thread_id": tid, "created_at": "", "title": f"会话 {tid[:12]}"}
                )
                known.add(tid)
    except Exception:
        pass
    # 置顶会话优先，组内按创建时间倒序
    recs.sort(key=lambda r: (bool(r.get("pinned", False)), r.get("created_at", "")), reverse=True)
    return recs


def delete_session(thread_id: str) -> None:
    """删除一个会话：sessions.json 记录 + SQLite 中该线程的全部检查点/写入。"""
    recs = _load_sessions()
    _save_sessions([r for r in recs if r["thread_id"] != thread_id])
    # 该会话私有的文档/向量也一并清掉，不留孤儿数据
    try:
        for d in list_documents(thread_id):
            delete_document(d["name"], thread_id)
    except Exception:
        pass
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        try:
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        except Exception:
            pass
        conn.commit()
        conn.close()
    except Exception:
        pass


# ========== 会话操作标签（[备忘] / [PPT] / [天气] ...） ==========
def _detect_tags(texts) -> list:
    joined = " ".join(texts or [])
    rules = [
        ("备忘", ("备忘", "备忘录", "memo", "已写入备忘", "已记录")),
        ("PPT", ("ppt", "幻灯片", "pptx", "演示文稿")),
        ("天气", ("天气", "温度", "℃", "摄氏度", "预报", "紫外线", "风向", "风速")),
        ("待办", ("待办", "todo", "任务清单")),
        ("提醒", ("提醒", "定时", "明早", "八点", "九点", "每天")),
        ("搜索", ("搜索", "查一下", "联网")),
        ("文档", ("文档", "知识库", "上传")),
        ("文件", (".txt", ".docx", ".md", "已保存", "写入文件")),
    ]
    tags = []
    for tag, kws in rules:
        if any(k.lower() in joined.lower() for k in kws):
            tags.append(tag)
    return tags


def update_session_tags(thread_id: str, texts) -> None:
    """根据一轮对话产生的文本，给会话打上操作标签（去重合并）。"""
    tags = _detect_tags(texts)
    if not tags:
        return
    recs = _load_sessions()
    for r in recs:
        if r["thread_id"] == thread_id:
            cur = set(r.get("tags", []))
            cur.update(tags)
            r["tags"] = sorted(cur)
            _save_sessions(recs)
            return


def load_conversation(graph, thread_id: str) -> list:
    """从持久化数据库读回某会话的问答对，供界面回访展示（过滤内部节点消息）。"""
    try:
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        msgs = state.values.get("messages", [])
    except Exception:
        return []
    pairs, user, ans = [], None, []
    for m in msgs:
        if isinstance(m, HumanMessage):
            if user is not None:
                pairs.append((user, ans[-1] if ans else None))
            user, ans = m.content, []
        elif isinstance(m, AIMessage):
            c = (m.content or "").strip()
            if not c or c.startswith("[规划]") or c.startswith("[评审]"):
                continue
            if user is not None:
                ans.append(c)
    if user is not None:
        pairs.append((user, ans[-1] if ans else None))
    out = []
    for u, a in pairs:
        out.append({"role": "user", "content": u})
        if a:
            out.append({"role": "assistant", "content": a})
    return out


def prune_checkpoints(keep: int = 3, vacuum: bool = True) -> int:
    """每个会话只保留最近 keep 个 checkpoint，回收数据库体积。返回删除的 checkpoint 数。
    本应用不用“时间回溯”，每个 checkpoint 又都存完整状态，不清理会越长越大。
    请在 Streamlit 停止后手动运行：python -c "import main; print(main.prune_checkpoints())" """
    conn = sqlite3.connect(DB_FILE)
    try:
        before = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        conn.execute(
            """DELETE FROM checkpoints WHERE rowid IN (
                 SELECT rowid FROM (
                   SELECT rowid, ROW_NUMBER() OVER (
                     PARTITION BY thread_id, checkpoint_ns ORDER BY checkpoint_id DESC) AS rn
                   FROM checkpoints) WHERE rn > ?)""",
            (keep,),
        )
        conn.execute(
            """DELETE FROM writes WHERE NOT EXISTS (
                 SELECT 1 FROM checkpoints c WHERE c.thread_id = writes.thread_id
                   AND c.checkpoint_ns = writes.checkpoint_ns
                   AND c.checkpoint_id = writes.checkpoint_id)"""
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        if vacuum:
            conn.execute("VACUUM")
        return before - after
    finally:
        conn.close()


# ========== 入口 ==========
if __name__ == "__main__":
    graph = get_graph()
    print("=" * 50)
    print("个人委托智能体已启动（DeepSeek 规划/执行 + Qwen 评审 + SQLite 持久化）")
    print("输入任务委托给智能体；输入 history 查看记忆；输入 quit 退出")
    print("=" * 50)

    while True:
        task = input("\n>>> 委托任务：").strip()
        if task.lower() in ("quit", "exit", "q"):
            break
        if task.lower() == "history":
            conn = sqlite3.connect(DB_FILE)
            rows = conn.execute(
                "SELECT checkpoint FROM checkpoints ORDER BY checkpoint_id LIMIT 3"
            ).fetchall()
            print(f"数据库已有 {len(rows)} 个检查点（可通过 thread_id 恢复）")
            conn.close()
            continue

        # 同一 thread_id 多次委托 => 记忆累积、可续接
        config = {"configurable": {"thread_id": "user-main"}}
        result = graph.invoke({"messages": [HumanMessage(content=task)]}, config=config)

        print("\n" + "=" * 50)
        print("最终回复：", result["execution_result"])
        print("-" * 50)
        print("评审意见：", result["review"])
        print("=" * 50)
