# -*- coding: utf-8 -*-
"""
个人委托智能体 - Streamlit 界面
运行：python -m streamlit run a.py
"""
import html
import requests
import streamlit as st
import time
from datetime import date, datetime
from langchain_core.messages import HumanMessage

import main

st.set_page_config(page_title="个人委托智能体", page_icon="🤖", layout="wide")

st.title("🤖 个人委托智能体")
st.caption("LangGraph 状态图 · DeepSeek 规划/执行 + Qwen 评审 · SQLite 持久化 · 断点续接")

# 启动后台调度线程（幂等）：定时提醒/自动任务到点自动触发
# （同时顺带启动本地提醒 HTTP 服务：127.0.0.1:8600，供前端 JS 轮询）
main.start_scheduler()

# ---------- 定时提醒弹窗（无 iframe / 无 fragment 方案，根治 DOM 竞态） ----------
# 说明：Streamlit 1.65 下 st.components.html(iframe) 每次 rerun 重建报 removeChild；
# st.fragment(run_every=5) 定时刷新报 insertBefore（均实测确认）；st.html 的 script 不执行。
# 因此采用最稳方案：每次页面 rerun（用户点会话/发消息等操作）时检查本地提醒服务(8600)，
# 有到点提醒则用 st.dialog 弹出模态框，必须点「确认」（ack）才关闭。
ALERT_API = "http://127.0.0.1:8600"


@st.dialog("⏰ 定时提醒", width="small")
def alert_dialog(a):
    st.markdown(f"**{a.get('content', '')}**")
    st.caption("触发时间：" + a.get("fired_at", ""))
    if st.button("确认"):
        try:
            requests.get(f"{ALERT_API}/ack?id={a['id']}", timeout=3)
        except Exception:
            pass
        st.session_state["__alert_dialog_open"] = None
        st.rerun()


def _check_alerts():
    try:
        r = requests.get(f"{ALERT_API}/alerts", timeout=3)
        return r.json() if r.ok else []
    except Exception:
        return []


if "__alert_dialog_open" not in st.session_state:
    st.session_state["__alert_dialog_open"] = None

if not st.session_state["__alert_dialog_open"]:
    _alerts = _check_alerts()
    if _alerts:
        st.session_state["__alert_dialog_open"] = _alerts[0]["id"]
        alert_dialog(_alerts[0])
        # 关键：弹窗时立即停止本轮渲染，避免"弹窗 + 消息/页面更新"并发导致
        # Streamlit 1.65 的 removeChild/insertBefore 竞态。确认关闭后才继续渲染。
        st.stop()


@st.cache_resource
def get_graph():
    return main.get_graph()


graph = get_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user-main"
if "history" not in st.session_state:
    st.session_state.history = []
if "manage_mode" not in st.session_state:
    st.session_state.manage_mode = False
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None
if "todo_manage" not in st.session_state:
    st.session_state.todo_manage = False
if "pending_todo_delete" not in st.session_state:
    st.session_state.pending_todo_delete = None
if "pending_todo_clear" not in st.session_state:
    st.session_state.pending_todo_clear = None

# 历史会话改为页面内切换（不改变 URL，不跳新页面）

CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}


# 新开会话按钮美化
st.markdown(
    """
    <style>
    button.st-key-new_conv_btn {
        padding: 0.85rem 1.2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #1677ff 0%, #2f9bff 55%, #45b0ff 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(22, 119, 255, 0.35) !important;
        letter-spacing: 1px !important;
        transition: all 0.15s ease !important;
    }
    button.st-key-new_conv_btn:hover {
        box-shadow: 0 5px 16px rgba(22, 119, 255, 0.55) !important;
        transform: translateY(-1px) !important;
    }
    button.st-key-new_conv_btn:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 6px rgba(22, 119, 255, 0.35) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("💬 会话控制")
    if st.button("🆕 新开会话", key="new_conv_btn", type="primary", use_container_width=True):
        new_tid = f"user-{int(time.time())}"
        st.session_state.thread_id = new_tid
        st.session_state.history = []
        main.register_session(new_tid, "")  # 立即可在历史列表看到「新会话」
        st.success("已新开会话（旧会话保存在数据库中，可在下方回访）")
        st.rerun()
    st.caption("💡 当前会话会持续累积记忆，直到你点击\"新开会话\"。")

    st.divider()
    manage = st.session_state.manage_mode
    col_t1, col_t2 = st.columns([0.7, 0.3])
    with col_t1:
        st.subheader("🗂️ 历史会话（回访）")
    with col_t2:
        if st.button("完成" if manage else "管理", key="manage_btn", use_container_width=True):
            st.session_state.manage_mode = not manage
            st.rerun()

    sessions = main.list_sessions()
    current_tid = st.session_state.thread_id

    def _shorten(name, limit=14):
        name = (name or "").strip()
        return name if len(name) <= limit else name[:limit] + "..."

    today = date.today()

    def _group_label(ts):
        """时间分组：今天 / 昨天 / 7天内 / 30天内 / 更早按月"""
        if not ts:
            return "更早"
        try:
            d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
        except ValueError:
            return "更早"
        diff = (today - d).days
        if diff <= 0:
            return "今天"
        if diff == 1:
            return "昨天"
        if diff < 7:
            return "7天内"
        if diff < 30:
            return "30天内"
        return f"{d.year}年{d.month}月"

    # 会话列表：无边框纯文本行 + 条目间分隔线 + 悬停整行变深
    st.markdown(
        """
        <style>
        [data-testid^="stSidebar"] button {
            width: 100% !important;
            justify-content: flex-start !important;
            text-align: left !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            border-radius: 0;
            padding: 8px 10px !important;
            margin: 0;
            white-space: normal !important;
            height: auto !important;
            display: flex;
            color: inherit !important;
            border-bottom: 1px solid #ECECEC !important;
        }
        [data-testid^="stSidebar"] button:hover,
        [data-testid^="stSidebar"] button:focus-visible {
            background: #E3E3E3 !important;
            border-bottom: 1px solid #ECECEC !important;
        }
        [data-testid^="stSidebar"] button p {
            margin: 0;
            line-height: 1.5;
            width: 100%;
            text-align: left !important;
        }
        [data-testid^="stSidebar"] button * {
            text-align: left !important;
        }
        /* 侧边栏全部元素统一左对齐（按钮/文本/标题/输入框/提示框等） */
        [data-testid^="stSidebar"] * {
            text-align: left !important;
        }
        [data-testid^="stSidebar"] button,
        [data-testid^="stSidebar"] button * {
            justify-content: flex-start !important;
        }
        [data-testid^="baseButton-sw_"] {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            border-radius: 0;
            padding: 8px 10px !important;
            margin: 0;
            white-space: normal !important;
            height: auto !important;
            border-bottom: 1px solid #ECECEC !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # DS 风格搜索框：按标题过滤
    search_q = st.text_input(
        "搜索对话内容", key="sess_search", label_visibility="collapsed",
        placeholder="搜索对话内容...",
    ).strip().lower()
    if search_q:
        sessions = [
            s for s in sessions
            if search_q in (s.get("title") or "").lower()
        ]

    groups = {}
    for s in sessions:
        g = _group_label(s.get("created_at") or "")
        groups.setdefault(g, []).append(s)

    if not groups:
        st.caption("_暂无历史会话_")
    else:
        for g, items in groups.items():
            st.markdown(
                f"<div style='color:#8A8A8A;font-size:12px;margin:10px 2px 2px 2px;'>{g}</div>",
                unsafe_allow_html=True,
            )
            for s in items:
                tid = s["thread_id"]
                t = _shorten(s.get("title") or "会话")
                # 会话行只显示标题（豆包风格），标签不再挤在标题里
                head = f"{'● ' if tid == current_tid else ''}💬 "
                label = f"{head}{t}"
                if manage:
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        if st.button(label, key=f"d_{tid}", help="点击切换会话", width="stretch"):
                            if tid != st.session_state.thread_id:
                                st.session_state.thread_id = tid
                                st.session_state.history = main.load_conversation(graph, tid)
                            st.rerun()
                    with c2:
                        if st.button("🗑️", key=f"dd_{tid}", help="删除该会话", use_container_width=True):
                            st.session_state.pending_delete = tid
                            st.rerun()
                    # 删除确认：显示在被删会话正下方
                    if st.session_state.get("pending_delete") == tid:
                        info = next((x for x in sessions if x["thread_id"] == tid), None)
                        name = _shorten((info or {}).get("title") or "该会话")
                        st.warning(f"是否删除「{name}」？")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("确认删除", key=f"confirm_del_{tid}", use_container_width=True):
                                main.delete_session(tid)
                                if tid == current_tid:
                                    st.session_state.thread_id = f"user-{int(time.time())}"
                                    st.session_state.history = []
                                st.session_state.pending_delete = None
                                st.rerun()
                        with c2:
                            if st.button("取消", key=f"cancel_del_{tid}", use_container_width=True):
                                st.session_state.pending_delete = None
                                st.rerun()
                else:
                    if st.button(label, key=f"sw_{tid}", width="stretch"):
                        if tid != st.session_state.thread_id:
                            st.session_state.thread_id = tid
                            st.session_state.history = main.load_conversation(graph, tid)
                        st.rerun()

    st.divider()
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.subheader("📋 待办")
    with col_t2:
        todo_manage = st.session_state.get("todo_manage", False)
        if st.button(
            "完成" if todo_manage else "管理",
            key="todo_manage_btn",
            use_container_width=True,
        ):
            st.session_state.todo_manage = not todo_manage
            st.rerun()

    todo_items = main.todo_items()
    if not todo_items:
        st.info("还没有待办，委托我记一条吧")
    elif todo_manage:
        # 管理模式：删除全部入口（带二次确认）
        if st.button("🗑️ 删除全部待办", key="clear_todos_btn", use_container_width=True):
            st.session_state.pending_todo_clear = True
            st.rerun()
        if st.session_state.get("pending_todo_clear"):
            st.warning(f"是否删除全部 {len(todo_items)} 条待办？此操作不可恢复。")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("确认全部删除", key="confirm_clear_todos", use_container_width=True):
                    main.clear_todos()
                    st.session_state.pending_todo_clear = None
                    st.session_state.todo_manage = False
                    st.rerun()
            with c2:
                if st.button("取消", key="cancel_clear_todos", use_container_width=True):
                    st.session_state.pending_todo_clear = None
                    st.rerun()
        st.divider()
        # 管理模式：逐条删除
        for i, t in enumerate(todo_items, 1):
            c1, c2 = st.columns([5, 1])
            with c1:
                mark = "✅" if t.get("done") else "⬜"
                st.markdown(f"{mark} {t.get('content', '')}")
            with c2:
                if st.button("🗑️", key=f"tdel_{i}", help=f"删除第 {i} 条"):
                    st.session_state.pending_todo_delete = i
                    st.rerun()
        pd_t = st.session_state.get("pending_todo_delete")
        if pd_t:
            target = (
                todo_items[pd_t - 1].get("content", "")
                if 1 <= pd_t <= len(todo_items)
                else ""
            )
            preview = target if len(target) <= 20 else target[:20] + "..."
            st.warning(f"是否删除这条待办？\n\n`{preview}`")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("确认删除", key="confirm_todo_del", use_container_width=True):
                    main.delete_todo(pd_t)
                    st.session_state.pending_todo_delete = None
                    st.rerun()
            with c2:
                if st.button("取消", key="cancel_todo_del", use_container_width=True):
                    st.session_state.pending_todo_delete = None
                    st.rerun()
    else:
        # 默认模式：未完成在前，右侧一键完成；已完成置灰展示
        pending = [(i + 1, t) for i, t in enumerate(todo_items) if not t.get("done")]
        done = [(i + 1, t) for i, t in enumerate(todo_items) if t.get("done")]
        for i, t in pending:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"⬜ {t.get('content', '')}")
            with c2:
                if st.button("✔", key=f"tdone_{i}", help="标记完成"):
                    main.complete_todo.invoke({"index": i})
                    st.rerun()
        for _i, t in done:
            st.markdown(f"~~✅ {t.get('content', '')}~~")

    st.divider()
    # ---------- 任务计划（拆解 + 进度跟踪） ----------
    c_p1, c_p2 = st.columns([3, 1])
    with c_p1:
        st.subheader("🎯 任务计划")
    with c_p2:
        if st.button("新建", key="plan_new_btn", use_container_width=True):
            st.session_state.plan_creating = not st.session_state.get("plan_creating", False)
            st.rerun()
    if st.session_state.get("plan_creating"):
        with st.container():
            _pt = st.text_input("委托一个长期任务，例如：准备考研复试", key="plan_task_input")
            _pd = st.text_input("截止日期（可选），例如：2026-12-20", key="plan_deadline_input")
            if st.button("拆解并创建", key="plan_create_btn", use_container_width=True):
                if _pt.strip():
                    try:
                        _msg = main.create_plan.invoke({"task": _pt.strip(), "deadline": _pd.strip()})
                        st.success("✔ " + _msg.splitlines()[0])
                        st.session_state.plan_creating = False
                        st.rerun()
                    except Exception as _e:
                        st.error(f"创建失败：{_e}")
                else:
                    st.warning("请先输入任务描述")
        st.divider()

    plans = main.plan_items()
    if not plans:
        st.caption("还没有任务计划。跟我说「帮我规划备考」，我会自动拆解成步骤")
    for p in plans:
        _pid = p["id"]
        _subs = p.get("subtasks", [])
        _done_n = sum(1 for s in _subs if s.get("status") == "已完成")
        _total = len(_subs)
        st.markdown(
            f"**{_shorten(p.get('task') or '未命名任务', 24)}**"
            + (f"　⏰ 截止 {p['deadline']}" if p.get("deadline") else "")
            + f"　`{_done_n}/{_total}`"
        )
        if _total:
            st.progress(_done_n / _total)
        for _si, _s in enumerate(_subs, 1):
            _st = _s.get("status", "未开始")
            _icon = {"未开始": "⬜", "进行中": "🔄", "已完成": "✅"}.get(_st, "⬜")
            _s1, _s2 = st.columns([5, 1])
            with _s1:
                st.markdown(f"{_icon} {_s.get('content', '')}")
            with _s2:
                _next_st = {"未开始": "进行中", "进行中": "已完成", "已完成": "未开始"}.get(_st, "进行中")
                if st.button(
                    "推进" if _st != "已完成" else "回退",
                    key=f"ps_{_pid}_{_si}",
                    help=f"当前：{_st} → 点击变{_next_st}",
                    use_container_width=True,
                ):
                    main.update_subtask.invoke({"plan_id": _pid, "index": _si, "status": _next_st})
                    st.rerun()
        _c1, _c2 = st.columns([5, 1])
        with _c1:
            st.caption(f"创建于 {p.get('created', '')}")
        with _c2:
            if st.button("🗑️", key=f"pdel_{_pid}", help="删除该计划", use_container_width=True):
                st.session_state.pending_plan_delete = _pid
                st.rerun()
        if st.session_state.get("pending_plan_delete") == _pid:
            st.warning(f"是否删除计划「{_shorten(p.get('task') or '', 20)}」？")
            _d1, _d2 = st.columns(2)
            with _d1:
                if st.button("确认删除", key=f"confirm_pdel_{_pid}", use_container_width=True):
                    main.delete_plan.invoke({"plan_id": _pid})
                    st.session_state.pending_plan_delete = None
                    st.rerun()
            with _d2:
                if st.button("取消", key=f"cancel_pdel_{_pid}", use_container_width=True):
                    st.session_state.pending_plan_delete = None
                    st.rerun()
        st.divider()
    with st.expander("📅 最近日报", expanded=False):
        from pathlib import Path as _P
        import os as _os

        _reports = sorted(
            _P(_os.path.expanduser("~/Desktop")).glob("日报_*.md"),
            key=lambda p: p.name,
            reverse=True,
        )
        if _reports:
            _latest = _reports[0]
            st.caption(f"📄 {_latest.name}")
            _body = _latest.read_text(encoding="utf-8").strip()
            st.markdown(_body[:1500] if len(_body) > 1500 else _body)
            if len(_body) > 1500:
                st.caption("…（内容较长，已截断显示）")
        else:
            st.info("还没有日报。跟我说「生成今日日报」，或「每天早上 8 点生成日报」")

    st.divider()
    with st.expander("📄 文档（当前会话私有）", expanded=False):
        if "kb_processed" not in st.session_state:
            st.session_state.kb_processed = {}
        up = st.file_uploader(
            "上传文档（txt/md/docx/pdf）", type=["txt", "md", "docx", "pdf"], key="kb_up"
        )
        if up is not None:
            _fkey = f"{st.session_state.thread_id}|{up.name}|{up.size}"
            if st.session_state.kb_processed.get(_fkey) is True:
                st.success(f"《{up.name}》已读入，直接问我即可")
            else:
                try:
                    _msg = main.add_document(
                        up.name, up.getvalue(), thread_id=st.session_state.thread_id
                    )
                    st.session_state.kb_processed[_fkey] = True
                    st.success("✔ 已读入，现在可以直接问我")
                except Exception as _e:
                    st.error(f"{type(_e).__name__}: {_e}")
        docs = main.list_documents(thread_id=st.session_state.thread_id)
        if docs:
            st.caption("本会话文档")
            for i, d in enumerate(docs):
                c1, c2 = st.columns([0.72, 0.28])
                with c1:
                    st.markdown(f"{d['name']} · {d['chunk_count']}片段")
                with c2:
                    if st.button("删除", key=f"kb_{i}", use_container_width=True):
                        st.success(
                            main.delete_document(
                                d["name"], thread_id=st.session_state.thread_id
                            )
                        )
                        st.rerun()
        else:
            st.caption("_本会话暂无文档（各会话独立私有）_")

# ---------- 消息气泡（替代 st.chat_message，规避 1.65 的 removeChild 竞态） ----------
def _bubble(content):
    safe = html.escape(content or "")
    st.markdown(
        f'<div style="display:flex;justify-content:flex-end;margin:6px 0;">'
        f'<div style="max-width:88%;background:#1677ff;color:#fff;padding:10px 14px;'
        f'border-radius:14px;font-size:15px;line-height:1.6;white-space:pre-wrap;'
        f'word-break:break-word;">{safe}</div></div>',
        unsafe_allow_html=True,
    )


# 首次进入：若当前会话有历史，从数据库加载展示
if not st.session_state.history:
    loaded = main.load_conversation(graph, st.session_state.thread_id)
    if loaded:
        st.session_state.history = loaded

for msg in st.session_state.history:
    # 不用 st.chat_message（1.65 下消息数量变化会触发 removeChild 竞态），
    # 用户消息用自定义气泡，助手消息直接 markdown 渲染
    if msg["role"] == "user":
        _bubble(msg["content"])
    else:
        st.markdown(msg["content"])

# ---------- 委托输入 ----------
task = st.chat_input("委托一个任务，例如：记个待办：周五下午开组会")

if task:
    st.session_state.history.append({"role": "user", "content": task})
    _bubble(task)

    # 完全同步执行：不渲染任何动态组件（无 spinner/empty/chat_message），
    # 执行期间 Streamlit 自带 Running 指示；结束后一次性渲染，杜绝 removeChild 竞态。
    execution_text = ""
    plan_text = ""
    trace_text = ""
    review_text = ""
    try:
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=task)]},
            config=CONFIG,
            stream_mode="updates",
        ):
            for node, data in chunk.items():
                if node == "planner" and data.get("plan"):
                    plan_text = data["plan"]

                elif node == "executor":
                    trace_text = data.get("execution_trace", "")
                    execution_text = data.get("execution_result", "") or execution_text

                elif node == "critic" and data.get("review"):
                    review_text = data["review"]

    except Exception as e:
        st.error(f"❌ 委托处理出错：{type(e).__name__}: {e}")

    # 一次性渲染结果（无并发 DOM 更新，彻底规避 removeChild）
    st.markdown("✅ **评审完成**")
    if plan_text:
        with st.expander("📋 规划方案", expanded=False):
            st.markdown(plan_text)
    if trace_text:
        with st.expander("🔍 工具调用证据链", expanded=False):
            st.code(trace_text)
    if review_text:
        with st.expander("🔍 评审意见（Qwen 独立校验）", expanded=False):
            st.markdown(review_text)
    # 只把最终结果直接展示出来，过程全在折叠下拉里
    st.markdown(execution_text or "（无执行结果）")

    st.session_state.history.append(
        {"role": "assistant", "content": execution_text or "（无执行结果）"}
    )
    main.register_session(st.session_state.thread_id, task)
    main.update_session_tags(
        st.session_state.thread_id, [task, execution_text or ""]
    )
