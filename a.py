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

st.set_page_config(page_title="眯哥委托智能体", page_icon="🤖", layout="wide")

_LOGO_B64 = "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1pVXfJRz9rGPfGT9pqknPfWCX//y2pdf8ooPvf/Dn7jyE2+79FJqHMCy6TKqDH/16q/tffjAYHMaCQaMcTUyNpz1OvBTxvS0WGz3nAQqttLWiVAspzkLh40BJm6pzfjkZFKt5L0OEPV9igV4XRTHXATvfdd7r/nWt177769541vfxEQAY1pl78Zf3Vip1bAbpaOkxljCWDA3+9Dvf/Or4aFmFAdFWmR55pgdHBgYGpm4/967fnfjb3/xy99c/oH/PPW0J5liwTHIQL725asDFfXS5Wf+wzMPH9r/xz/+cf369bNzc1d88H2dduvgwcM33/zfD+/f8+nPfqEo2n05GNlud9/8ujckcXTueef8w3Nf8Ksbfq3LUgZyw4ZNX7n6i1GkTty+nTFhDWSh23ac8uSnPOWrV33pFS992TnnPeNnP/6f5cWloaHmsy94zste8aqy6NEQOhS8BrC6PM8X5xdGR0Y/9fGPPf5JT/x/b3j74NAAY/WNmza/8pKXfv97188vzL3p0rdc/Px/XlleHh4aNMY+8QlPvuQV/2/dhg2Pe8JT773nTz/43neFFM3BwbvuvP1nP/vZZz7/ufe88/Vf/8qXhppDSTXZceL251/8AhpYt9pU6sO33fKHz3z289/+zrecS7ETB+XcusmxLOviDJPu5WKxA61RMDLDgBuykhbBcKSxXjam3LxxAxeRYx1Uv4CKrdRltT64tLh82aVv/8n1P3r9m17/uje9Pusug5saGb/2+z/ct+9goYsTTtwWxWP33nPzz3/68zCMlZTT6zc7a6QKVzvLJfDpWK1RV4GoNepJpXLk0KHL3/Wuk3efetHzLzhp11O++NkP3XPXvUqF6zdtOOe88z73qf9+ytlnH3/88T/43g/27HnolNNOe/W//fuehx88bstGrXvMWOi6IX4RxmHaSe+7657f/vo3s3OzkLpZO1Ad+NrXv3777Xdc9LwLtWFlWQRhCPVJd+HFL3nRn/7429mZua986eqp6ekrP/3f6zesGx8ZhQnbEib0CDQTQhprKtXG9hNPvPee7z79/Be/9R3v0WUvS1fiJLr15j9aZkbGxn9/059e/NLZS9952U033XT+M5752U9+/IfXXnfRC//pxZf8v49/9PJvfPmr1skdO7e/+e2XfuELn9u69fgrPvCh22+9+f0f+nCe96764uc/fOVHxsY3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"

st.markdown(
    '<div style="display:flex;align-items:center;gap:10px;">'
    f'<img src="data:image/png;base64,{_LOGO_B64}" style="height:42px;width:auto;border-radius:8px;flex-shrink:0;">'
    '<span style="font-size:34px;font-weight:600;line-height:1.3;">眯哥委托智能体</span>'
    "</div>",
    unsafe_allow_html=True,
)
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


with st.sidebar:
    st.subheader("💬 会话控制")
    if st.button("新开会话", icon=":material/add_circle:", key="new_conv_btn", type="primary", use_container_width=True):
        new_tid = f"user-{int(time.time())}"
        st.session_state.thread_id = new_tid
        st.session_state.history = []
        main.register_session(new_tid, "")  # 立即可在历史列表看到「新会话」
        st.rerun()
    st.caption("💡 会话会持续累积记忆，不同会话记忆不共存。")

    # tabs 分区：未激活 tab 前端不渲染，大幅减小 rerun 时的 DOM 组件树，
    # 根治 Streamlit 1.65 在大型侧边栏下的 removeChild 竞态
    tab_conv, tab_todo, tab_plan, tab_doc = st.tabs(
        ["💬 会话", "📋 待办", "🎯 计划", "📄 文档"]
    )

    with tab_conv:
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

        pinned_items = [s for s in sessions if s.get("pinned")]
        normal_sessions = [s for s in sessions if not s.get("pinned")]

        def _render_group(g, items):
            st.markdown(
                f"<div style='color:#8A8A8A;font-size:12px;margin:10px 2px 2px 2px;'>{g}</div>",
                unsafe_allow_html=True,
            )
            for s in items:
                tid = s["thread_id"]
                t = _shorten(s.get("title") or "会话")
                head = f"{'● ' if tid == current_tid else ''}{'📌 ' if s.get('pinned') else ''}💬 "
                label = f"{head}{t}"
                if manage:
                    c1, c2, c3 = st.columns([4, 1, 1])
                    with c1:
                        if st.button(label, key=f"d_{tid}", help="点击切换会话", width="stretch"):
                            if tid != st.session_state.thread_id:
                                st.session_state.thread_id = tid
                                st.session_state.history = main.load_conversation(graph, tid)
                            st.rerun()
                    with c2:
                        if st.button("📍" if s.get("pinned") else "📌", key=f"pin_{tid}", help="置顶/取消置顶", use_container_width=True):
                            main.set_pinned(tid, not s.get("pinned", False))
                            st.rerun()
                    with c3:
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

        if not sessions:
            st.caption("_暂无历史会话_")
        else:
            if pinned_items:
                _render_group("📌 置顶", pinned_items)
            groups = {}
            for s in normal_sessions:
                g = _group_label(s.get("created_at") or "")
                groups.setdefault(g, []).append(s)
            for g, items in groups.items():
                _render_group(g, items)

    with tab_todo:
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

    with tab_plan:
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

    with tab_doc:
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

# 新会话欢迎语（含文档能力引导）
if not st.session_state.history:
    st.markdown(
        "你好！👋 我是你的执行专员，随时可以帮你处理这些事：\n\n"
        "- 🗓️ 生成今日日报：汇总天气、待办、备忘和知识库沉淀\n"
        "- ⏰ 设定提醒：一次性 / 每天 / 每周定时提醒\n"
        "- 📝 记录备忘、添加待办\n"
        "- 📄 分析文档：上传 PDF / Word / 图片，帮你总结、问答、提炼要点\n"
        "- 🗂️ 规划长期目标：备考、学习计划、项目推进等自动拆解\n"
        "- 🍽️ 生成菜单、查天气、联网搜索并沉淀笔记\n\n"
        "需要我做什么，直接说一声就行～"
    )

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

    # 思考中提示：静态渲染，执行完 rerun 刷新后自动消失（不触发前端竞态）
    st.markdown("🤔 **正在思考…**")
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
        execution_text = f"❌ 委托处理出错：{type(e).__name__}: {e}"

    st.session_state.history.append(
        {"role": "assistant", "content": execution_text or "（无执行结果）"}
    )
    main.register_session(st.session_state.thread_id, task)
    main.update_session_tags(
        st.session_state.thread_id, [task, execution_text or ""]
    )
    # 刷新页面：思考提示消失，答案随历史消息一次性渲染
    st.rerun()
