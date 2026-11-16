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
# 运行期数据目录：数据库/会话/计划/知识库/备忘都放这里。
# 本地默认就是项目根目录；容器部署时用 DATA_DIR 环境变量指向挂载卷，实现持久化。
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = DATA_DIR / "agent_memory.db"
MEMO_FILE = DATA_DIR / "memos.md"

if not DEEPSEEK_KEY or not QWEN_KEY:
    raise SystemExit("请在 .env 中配置 DEEPSEEK_API_KEY 和 QWEN_API_KEY")


# ========== 模型工厂（异构：DeepSeek + Qwen） ==========
def deepseek_model():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.3,
    )


def qwen_model():
    return ChatOpenAI(
        model="qwen-plus",
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.2,
    )


# ========== 工具（执行节点可调用） ==========
@tool
def get_today() -> str:
    """获取当前精确时间（年月日 时:分 星期）。用户说'今天''明天''两小时后''X小时后'等相对时间时必须先调用本工具。"""
    from datetime import date, datetime, timedelta

    now = datetime.now()
    wd = "一二三四五六日"[now.weekday()]
    return (
        f"现在 {now.strftime('%Y-%m-%d %H:%M')} 星期{wd}；"
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


# ---------- 待办管理 ----------
TODO_FILE = DATA_DIR / "todos.json"


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
PLANS_FILE = DATA_DIR / "plans.json"


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

KB_DIR = DATA_DIR / "kb"
KB_CHUNKS = KB_DIR / "chunks.json"   # [{thread_id, doc, text}]
KB_VECS = KB_DIR / "vectors.npy"     # numpy 矩阵（行 = 片段）
KB_DOCS = KB_DIR / "docs.json"       # [{thread_id, name, chunk_count, created_at}]

# 当前执行会话（executor 节点设置，工具按此隔离知识库）
_ACTIVE_THREAD = None

EMBED_MODEL = "text-embedding-v3"
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


def _embed_texts(texts: list) -> list:
    """批量文本 → Qwen text-embedding-v3 向量（每次最多 10 条）。"""
    out = []
    for i in range(0, len(texts), 10):
        batch = texts[i : i + 10]
        resp = requests.post(
            EMBED_URL,
            headers={"Authorization": f"Bearer {QWEN_KEY}"},
            json={"model": EMBED_MODEL, "input": batch},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)
    return out


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
    KB_DIR.mkdir(parents=True, exist_ok=True)
    KB_CHUNKS.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    np.save(KB_VECS, vecs)
    KB_DOCS.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")


def add_document(filename: str, content: bytes, thread_id: str = "user-main") -> str:
    """解析一份上传的文档并入库（结构化解析 → 切片 → 向量化 → 追加存储），文档归属指定会话，仅该会话可见。

    - 支持 PDF / Word / Markdown / txt / 图片（图片与扫描页走 OCR）
    - 每个切片携带 page / heading / kind，供答案溯源到「第几页 · 哪一节」
    - 同名文档按内容哈希判重：内容未变直接跳过，避免重复消耗 embedding
    """
    import documents as docengine

    digest = docengine.sha256(content)
    chunks_old, vecs_old, docs = _load_kb()

    same = [d for d in docs if d["thread_id"] == thread_id and d["name"] == filename]
    if same and same[0].get("sha256") == digest:
        return f"《{filename}》内容未变更，跳过重复解析（已有 {same[0].get('chunk_count', 0)} 个片段）"

    blocks = docengine.extract_blocks(filename, content)
    new_chunks = docengine.chunk_blocks(blocks)
    if not new_chunks:
        raise ValueError("文档内容为空或无法提取文字")

    vecs_new = np.array(_embed_texts([c.text for c in new_chunks]), dtype="float32")

    # 同名文档覆盖：先摘掉旧片段，再追加新片段（避免同一份文件多版本共存）
    keep = [i for i, c in enumerate(chunks_old) if not (c["thread_id"] == thread_id and c["doc"] == filename)]
    chunks_old = [chunks_old[i] for i in keep]
    if len(vecs_old):
        vecs_old = vecs_old[keep]

    for i, c in enumerate(new_chunks):
        chunks_old.append({
            "thread_id": thread_id,
            "doc": filename,
            "idx": i,
            "text": c.text,
            "page": c.page,
            "heading": c.heading,
            "kind": c.kind,
        })

    vecs = np.vstack([vecs_old, vecs_new]) if len(vecs_old) else vecs_new
    docs = [d for d in docs if not (d["thread_id"] == thread_id and d["name"] == filename)]
    pages = sorted({c.page for c in new_chunks if c.page})
    docs.append({
        "thread_id": thread_id,
        "name": filename,
        "chunk_count": len(new_chunks),
        "page_count": len(pages),
        "table_count": sum(1 for c in new_chunks if c.kind == "table"),
        "ocr_count": sum(1 for c in new_chunks if c.kind == "image"),
        "sha256": digest,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_kb(chunks_old, vecs, docs)

    extra = []
    if pages:
        extra.append(f"{len(pages)} 页")
    if docs[-1]["table_count"]:
        extra.append(f"{docs[-1]['table_count']} 个表格")
    if docs[-1]["ocr_count"]:
        extra.append(f"{docs[-1]['ocr_count']} 段 OCR")
    suffix = ("，含 " + "、".join(extra)) if extra else ""
    return f"已读入《{filename}》：{len(new_chunks)} 个知识片段{suffix}，可以直接问我"


def list_documents(thread_id: str = None) -> list:
    """列出文档；thread_id 为空返回全部（管理用），否则只返回该会话的。"""
    _, _, docs = _load_kb()
    if thread_id is None:
        return docs
    return [d for d in docs if d["thread_id"] == thread_id]


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


def _bm25_scores(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """字符二元组 BM25：解决「合同编号」「A-2024-001」这类精确词被向量稀释的问题。

    中文没有天然分词，用二元字组（bigram）当 term 足够稳，且不引入分词依赖。
    """
    def grams(s: str) -> list[str]:
        s = re.sub(r"\s+", "", s)
        if len(s) < 2:
            return [s] if s else []
        return [s[i:i + 2] for i in range(len(s) - 1)]

    q_grams = grams(query)
    if not q_grams or not texts:
        return np.zeros(len(texts), dtype="float32")

    doc_grams = [grams(t) for t in texts]
    n = len(doc_grams)
    avg_len = sum(len(g) for g in doc_grams) / max(n, 1) or 1.0

    df: dict[str, int] = {}
    for g in doc_grams:
        for term in set(g):
            df[term] = df.get(term, 0) + 1

    import math
    idf = {t: math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in set(q_grams)}

    scores = np.zeros(n, dtype="float32")
    for i, g in enumerate(doc_grams):
        tf: dict[str, int] = {}
        for term in g:
            tf[term] = tf.get(term, 0) + 1
        dl = len(g)
        s = 0.0
        for term in q_grams:
            if term not in tf:
                continue
            f = tf[term]
            s += idf.get(term, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_len))
        scores[i] = s

    # 归一到 0~1，便于与余弦分加权
    mx = float(scores.max()) if scores.size else 0.0
    if mx > 0:
        scores = scores / mx
    return scores


def preview_document(name: str, thread_id: str = None, n: int = 4) -> list[dict]:
    """取某份文档的前 n 个切片，供界面预览解析效果（页码/章节/类型 + 正文）。"""
    chunks, _, _ = _load_kb()
    picked = [
        c for c in chunks
        if c["doc"] == name and (thread_id is None or c["thread_id"] == thread_id)
    ]
    return [
        {"page": c.get("page"), "heading": c.get("heading", ""),
         "kind": c.get("kind", "text"), "text": c.get("text", "")}
        for c in picked[:n]
    ]


def highlight_terms(text: str, query: str, max_marks: int = 8) -> str:
    """把 query 中与 text 重合的片段高亮（返回可交给 st.markdown 的 HTML）。

    中文没有空格分隔，「违约金是多少」不可能整串出现在正文里，
    因此按二元组扫描、合并连续命中，得到「违约金」这样的完整词。
    """
    import html as _html

    t = text or ""
    q = re.sub(r"\s+", "", query or "")
    if len(q) < 2 or not t:
        return _html.escape(t)

    qgrams = {q[i:i + 2] for i in range(len(q) - 1)}
    spans: list[tuple[int, int]] = []
    i, n = 0, len(t)
    while i < n - 1:
        if t[i:i + 2] in qgrams:
            j = i + 2
            while j < n - 1 and t[j:j + 2] in qgrams:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1

    if not spans:
        return _html.escape(t)

    out, last = [], 0
    for a, b in spans[:max_marks]:
        out.append(_html.escape(t[last:a]))
        out.append(f"<mark>{_html.escape(t[a:b])}</mark>")
        last = b
    out.append(_html.escape(t[last:]))
    return "".join(out)


def search_documents(query: str, top_k: int = 5, thread_id: str = None,
                     min_score: float | None = None) -> list:
    """混合检索：向量余弦 + 字符 BM25 加权融合，按会话过滤，低于阈值不返回。

    纯向量检索对「违约金」「A-2024-001」这类精确词不敏感，
    因此用 BM25 补一路分数：final = 0.7 * cosine + 0.3 * bm25。
    阈值默认读环境变量 DOC_MIN_SCORE，未设置则 0.42，问不到就不硬答。
    """
    if min_score is None:
        try:
            min_score = float(os.getenv("DOC_MIN_SCORE", "0.42"))
        except ValueError:
            min_score = 0.42
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
    cosine = (vecs @ qv) / (norms * qn)

    # 混合检索：向量语义 + BM25 精确词，两路加权融合
    bm25 = _bm25_scores(query, [c["text"] for c in chunks])
    adjusted = 0.7 * cosine + 0.3 * bm25

    idx = np.argsort(-adjusted)[:top_k]
    return [
        {
            "doc": chunks[i]["doc"],
            "idx": chunks[i].get("idx", 0),
            "text": chunks[i]["text"],
            "page": chunks[i].get("page"),
            "heading": chunks[i].get("heading", ""),
            "kind": chunks[i].get("kind", "text"),
            "score": float(adjusted[i]),
            "cosine": float(cosine[i]),
            "bm25": float(bm25[i]),
        }
        for i in idx
        if float(adjusted[i]) >= min_score
    ]


_GLOBAL_Q = re.compile(r"(总结|概括|归纳|全文|整体|大纲|结构|主要(讲|说|内容)|讲了(什么|啥)|全文要点)")

# 最近一次检索命中的片段，供界面展示「引用原文」（不参与业务逻辑）
_LAST_HITS: list[dict] = []


def _thread_chunks(thread_id: str) -> list[dict]:
    """取出当前会话的全部片段（按页/顺序），供全局摘要使用。"""
    chunks, _, _ = _load_kb()
    return [c for c in chunks if c["thread_id"] == thread_id]


def summarize_document(query: str, thread_id: str, max_chunks: int = 60) -> str:
    """全局摘要（map-reduce）：先分块摘要，再归并成总述。

    top-k 检索只能覆盖文档的一小部分，问「总结全文」时必然漏内容，
    因此这类问题走「逐批摘要 → 汇总」的两段式流程。
    """
    import documents as docengine

    chunks = _thread_chunks(thread_id)
    if not chunks:
        return "本会话还没有上传文档，请先在侧边栏「文档」区上传后再让我总结。"
    chunks = chunks[:max_chunks]

    def label(c: dict) -> str:
        return docengine.citation(c)

    # map：每批 8 个片段做一次局部摘要
    batch_size = 8
    partials: list[str] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        body = "\n\n".join(f"[{label(c)}] {c['text']}" for c in batch)
        prompt = (
            "下面是某份文档的一部分内容。请提炼这段内容的关键信息（要点式，不超过 5 条），"
            "只依据给定内容，不要补充外部知识。\n\n" + body
        )
        resp = deepseek_model().invoke([HumanMessage(content=prompt)])
        partials.append(f"第 {start // batch_size + 1} 部分：\n{resp.content}")

    # reduce：汇总各部分，回答用户的全局问题
    merged = "\n\n".join(partials)
    prompt = (
        f"用户问题：{query}\n\n以下是文档各部分的摘要：\n{merged}\n\n"
        "请综合上述内容回答用户问题，给出结构化总述（可用小标题或分点）。"
        "结尾用一行注明依据的文档名。只依据给定内容，不要编造。"
    )
    resp = deepseek_model().invoke([HumanMessage(content=prompt)])
    docs = sorted({c["doc"] for c in chunks})
    return f"{resp.content}\n\n（依据：{'、'.join(f'《{d}》' for d in docs)}，共 {len(chunks)} 个片段）"


@tool
def extract_fields(fields: str) -> str:
    """从当前会话的文档中抽取指定字段并整理成表格。

    fields 用逗号分隔要抽取的字段，例如："甲方, 乙方, 合同金额, 签署日期, 违约金比例"。
    用户要求「提取/汇总/整理出关键信息」时要调用本工具，而不是直接看片段自己拼。
    """
    import documents as docengine

    thread = _ACTIVE_THREAD or "user-main"
    chunks = _thread_chunks(thread)
    if not chunks:
        return "本会话还没有上传文档。请先在侧边栏「文档」区上传（支持 PDF / Word / Markdown / 图片）后再抽取。"

    field_list = [f.strip() for f in re.split(r"[,，、;；]", fields) if f.strip()]
    if not field_list:
        return "请给出要抽取的字段，用逗号分隔，例如：甲方, 乙方, 金额"

    # 每个字段单独检索，保证长文档里不同位置的字段都能被覆盖到
    picked: dict[tuple, dict] = {}
    for f in field_list:
        for h in search_documents(f, top_k=2, thread_id=thread, min_score=0.30):
            picked[(h["doc"], h["idx"])] = h

    ctx = "\n\n".join(
        f"[{docengine.citation(h)}]\n{h['text']}"
        for h in list(picked.values())[:12]
    )
    prompt = (
        "你是信息抽取助手。请从下面的文档片段中抽取指定字段。\n"
        f"需要抽取的字段：{'、'.join(field_list)}\n\n"
        "要求：\n"
        "1. 只使用片段中出现的信息，找不到的字段填「未提及」，严禁编造；\n"
        "2. 输出 Markdown 表格，列为：字段 | 内容 | 出处；\n"
        "3. 出处写成「文档名 第 N 页」或「文档名 章节名」，取自片段方括号里的标注。\n\n"
        f"文档片段：\n{ctx}"
    )
    resp = deepseek_model().invoke([HumanMessage(content=prompt)])
    return resp.content


@tool
def compare_documents(doc_a: str, doc_b: str, aspect: str = "关键差异") -> str:
    """对比当前会话中的两份文档，输出差异表格。

    doc_a / doc_b 是文档名的关键字（不必写全名），aspect 是要对比的方面，例如"付款条款""责任划分"。
    """
    import documents as docengine

    thread = _ACTIVE_THREAD or "user-main"
    chunks = _thread_chunks(thread)
    if not chunks:
        return "本会话还没有上传文档，无法对比。"

    def pick(name: str, k: int = 6) -> list[dict]:
        return [c for c in chunks if name.lower() in c["doc"].lower()][:k]

    a_chunks, b_chunks = pick(doc_a), pick(doc_b)
    if not a_chunks:
        docs = sorted({c["doc"] for c in chunks})
        return f"找不到《{doc_a}》，当前会话文档：{'、'.join(docs)}"
    if not b_chunks:
        docs = sorted({c["doc"] for c in chunks})
        return f"找不到《{doc_b}》，当前会话文档：{'、'.join(docs)}"

    def block(cs: list[dict]) -> str:
        return "\n\n".join(f"[{docengine.citation(c)}]\n{c['text']}" for c in cs)

    prompt = (
        f"你是文档对比助手。请就「{aspect}」对比下面两份文档。\n\n"
        f"=== 文档 A（{a_chunks[0]['doc']}）===\n{block(a_chunks)}\n\n"
        f"=== 文档 B（{b_chunks[0]['doc']}）===\n{block(b_chunks)}\n\n"
        "要求：输出 Markdown 表格（列：对比项 | 文档 A | 文档 B），"
        "只依据给定内容，缺失的信息写「未提及」，最后用一句话总结主要差异。"
    )
    resp = deepseek_model().invoke([HumanMessage(content=prompt)])
    return resp.content


@tool
def ask_document(query: str) -> str:
    """基于当前会话已上传的文档回答用户问题（RAG 检索 + 生成）。

    用户在问文档/资料/文件里写了什么时必须先调用本工具；
    问「总结/概括/全文要点」这类全局问题时，工具内部会自动改用 map-reduce 摘要。
    """
    import documents as docengine

    thread = _ACTIVE_THREAD or "user-main"

    # 全局问题 → 两段式摘要，避免 top-k 覆盖不足
    if _GLOBAL_Q.search(query):
        return summarize_document(query, thread)

    hits = search_documents(query, thread_id=thread)
    if not hits:
        if not _thread_chunks(thread):
            return "本会话还没有上传文档。请先在侧边栏「文档」区上传（支持 PDF / Word / Markdown / 图片）后再提问。"
        return "知识库中没有检索到与这个问题相关的内容（相似度低于阈值）。可以换个说法，或确认相关文档是否已上传。"

    ctx = "\n\n".join(
        f"[{i+1}] {docengine.citation(h)}\n{h['text']}" for i, h in enumerate(hits)
    )
    _LAST_HITS[:] = hits
    prompt = (
        "你是文档分析助手。严格依据下面提供的文档片段回答用户问题；"
        "如果片段中没有相关信息，直接说明'知识库中没有相关内容'，不要编造。"
        "回答末尾用 [1][2] 标注引用的片段编号。\n\n"
        f"文档片段：\n{ctx}\n\n用户问题：{query}\n\n回答："
    )
    resp = deepseek_model().invoke([HumanMessage(content=prompt)])
    src = "、".join(dict.fromkeys(docengine.citation(h) for h in hits))
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
    ):
        try:
            conn.execute(col_sql)
        except sqlite3.OperationalError:
            pass
    return conn


def _parse_at(at: str, repeat: str):
    """把模型给的时间格式解析成 (datetime, 星期几, HH:MM)。"""
    if repeat == "once":
        return datetime.strptime(at.strip(), "%Y-%m-%d %H:%M"), None, None
    if repeat == "daily":
        hm = at.strip()
        datetime.strptime(hm, "%H:%M")  # 校验格式
        return None, None, hm
    if repeat == "weekly":
        parts = at.strip().split()
        if len(parts) != 2:
            raise ValueError("weekly 格式应为：周X HH:MM")
        wd = parts[0].lower().lstrip("周星期礼拜week")
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
    repeat=once 时 at 为具体时间 "YYYY-MM-DD HH:MM"（如 2026-10-13 08:00）；
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
        "SELECT id, content, at, repeat, action, action_arg FROM reminders"
    ).fetchall()
    for rid, content, at, repeat, action, action_arg in rows:
        if not _is_due(at, repeat, now):
            continue
        fired_at = now.strftime("%Y-%m-%d %H:%M")
        alert_text = f"⏰ {content}"
        if action == "daily_report":
            # 动作型提醒：到点真正执行“聚合日报”任务链
            try:
                report = generate_daily_report(city=action_arg or "")
                head, _, rest = report.partition("\n")
                preview = "\n".join(rest.split("\n")[:3])
                alert_text = f"📅 今日日报已生成\n{head}\n{preview}..."
            except Exception as e:
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
                "UPDATE reminders SET at=? WHERE id=?",
                (_advance_at(at, repeat), rid),
            )
    conn.commit()
    conn.close()


_scheduler_started = False


def start_scheduler():
    """启动后台调度线程（幂等）：每 30 秒检查一次到期提醒。"""
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True

    def _loop():
        while True:
            try:
                _check_due()
            except Exception:
                pass
            _time.sleep(30)

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
        self.send_header("Access-Control-Allow-Origin", "*")
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
            elif u.path == "/ack":
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
        add_document(filename, note.encode("utf-8"), thread_id=_ACTIVE_THREAD)
        return f"「{query}」搜索结果：\n\n{body}\n\n📚 已整理成《{filename}》存入知识库，下次可直接问我"
    except Exception as e:
        return f"「{query}」搜索结果：\n\n{body}\n\n（笔记入库失败：{e}）"


TOOLS = [
    get_today, write_memo, write_file, add_todo, list_todos, complete_todo,
    get_weather, ask_document, extract_fields, compare_documents,
    create_reminder, list_reminders,
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
    "只有在任务缺少「无法从工具或上下文获得」的关键参数时才追问，"
    "输出一行：'需要补充：<一句话具体问题>'。"
    "以下情况一律不要追问，直接给计划：时间和日期（有 get_today 工具）、天气（可查默认城市）、"
    "待办/计划/文档列表（有对应工具）、用户已经在上面说过的信息。"
)


def planner_node(state: State):
    user_task = _last_user_task(state)
    context = _build_context(state)
    prompt = f"对话历史：\n{context}\n\n当前任务：{user_task}"
    resp = deepseek_model().invoke([PLANNER_PROMPT, HumanMessage(content=prompt)])
    plan = (resp.content or "").strip()

    # 关键信息缺失 → 直接向用户澄清，不进入执行（避免瞎猜参数）
    if plan.startswith("需要补充") or "需要补充：" in plan[:12]:
        question = plan.split("：", 1)[-1].strip() if "：" in plan else plan
        return {
            "plan": plan,
            "execution_result": f"🤔 {question}",
            "messages": [AIMessage(content=f"🤔 {question}")],
        }
    return {"plan": plan, "messages": [AIMessage(content=f"[规划] {plan}")]}


def _plan_router(state: State) -> str:
    """规划阶段判断信息是否齐全：需要补充 → 直接结束并反问用户。"""
    plan = state.get("plan", "") or ""
    if plan.startswith("需要补充") or "需要补充：" in plan[:12]:
        return "end"
    return "executor"


# ========== 节点2：执行（DeepSeek + 工具循环） ==========
EXECUTOR_PROMPT = SystemMessage(
    "你是执行专员。严格按照计划调用可用工具完成任务："
    "1. 需要日期时先调 get_today；"
    "2. 需要记录备忘时调 write_memo；"
    "3. 用户询问已上传文档/知识库里的内容时，必须调 ask_document；"
    "3.1 用户要求从文档里提取/整理字段（如甲方、金额、期限）时，调 extract_fields；"
    "3.2 用户要求对比两份文档时，调 compare_documents；"
    "3.3 用户问「总结全文 / 概括要点」这类全局问题时，仍调 ask_document（内部会自动走分段摘要）；"
    "3.4 分析完成后用户要求导出/保存成文件时，把结论整理成 Markdown 再调 write_file（.docx / .pptx 会自动转换）；"
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
                    result = f"工具执行出错：{e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            trace.append(f"调用 {tc['name']}({tc['args']}) -> {result}")
    return last, messages, trace


def executor_node(state: State, config=None):
    global _ACTIVE_THREAD
    if config and config.get("configurable", {}).get("thread_id"):
        _ACTIVE_THREAD = config["configurable"]["thread_id"]
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
MAX_RETRY = 2


def _review_router(state: State) -> str:
    """评审说"需修正"且未超重做上限 → 回 executor 返工；否则结束。"""
    if "需修正" in state.get("review", "") and state.get("retry_count", 0) < MAX_RETRY:
        return "executor"
    return "end"


# ========== 构建状态图 ==========
def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)
    g.add_edge(START, "planner")
    # 规划后分流：信息不全 → 直接反问用户；齐全 → 进入执行
    g.add_conditional_edges("planner", _plan_router, {"executor": "executor", "end": END})
    g.add_edge("executor", "critic")
    # 评审"需修正"→ 回 executor 重做（自主决策循环）；通过 → 结束
    g.add_conditional_edges("critic", _review_router, {"executor": "executor", "end": END})
    return g.compile(checkpointer=checkpointer)


def get_graph():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_graph(checkpointer)


# ========== 会话管理（回访历史会话） ==========
SESSIONS_FILE = DATA_DIR / "sessions.json"


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
    recs.sort(key=lambda r: (not r.get("pinned", False), r.get("created_at", "")), reverse=True)
    return recs


def delete_session(thread_id: str) -> None:
    """删除一个会话：sessions.json 记录 + SQLite 中该线程的全部检查点/写入。"""
    recs = _load_sessions()
    _save_sessions([r for r in recs if r["thread_id"] != thread_id])
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
