# -*- coding: utf-8 -*-
"""本地存储：备忘、待办、计划、会话索引（JSON/文本文件，零外部依赖）。"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from runtime import call_llm, deepseek_model
from settings import *  # noqa: F401,F403

log = logging.getLogger("agent.store")

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
        resp = call_llm(deepseek_model(), [HumanMessage(content=prompt)])
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

__all__ = [n for n in list(globals()) if not n.startswith("__")]
