# -*- coding: utf-8 -*-
"""ReAct 终态信号（ASK / ABORT / CONTINUE）+ 重规划回边的离线断言。

跑：python test_react.py
全部用假模型驱动，不联网、不花钱。
"""
import os
import sys
import tempfile
from pathlib import Path

_TMP_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP_BASE, exist_ok=True)
_tmp = tempfile.mkdtemp(prefix="react-test-", dir=_TMP_BASE)
os.environ["GOALS_DB"] = str(Path(_tmp) / "goals.db")
os.environ["DATA_DIR"] = _tmp  # 会话/备忘等不污染开发目录
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import goals  # noqa: E402
import main  # noqa: E402

_passed, _failed = 0, 0


def ok(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def check(name):
    print(f"\n== {name} ==")


class FakeModel:
    """只要能被 bind_tools 调用即可，真正的回复由 fake call_llm 决定。"""

    def bind_tools(self, tools):
        return self


class Script:
    """按 what 关键字给出预设回复的假 call_llm。"""

    def __init__(self, planner=None, executor=None, critic=None):
        self.plan = list(planner or [])
        self.exec = list(executor or [])
        self.crit = list(critic or [])
        self.calls = []

    def __call__(self, model, messages, what="", *a, **kw):
        self.calls.append(what)
        if "规划" in what:
            return AIMessage(content=self.plan.pop(0))
        if "执行" in what:
            return self.exec.pop(0)
        if "评审" in what:
            return AIMessage(content=self.crit.pop(0))
        raise AssertionError(f"意外的 what：{what}")


def install(script):
    main.call_llm = script
    main.deepseek_model = lambda: FakeModel()
    main.qwen_model = lambda: FakeModel()


PLAN_JSON = '{"plan": "①查天气", "checks": ["给出温度数值"], "need_input": ""}'
PLAN_JSON_2 = '{"plan": "①换个城市查", "checks": ["给出温度数值"], "need_input": ""}'
PASS_JSON = '{"verdict": "pass", "failed": [], "reason": "已给出温度"}'

# ---------------- 1. 信号解析 ----------------
check("1. 终态信号解析")
ok("ASK 半角冒号", main._signal("ASK: 哪天的票？") == ("ASK", "哪天的票？"), main._signal("ASK: x"))
ok("ABORT 全角冒号", main._signal("ABORT：缺数据") == ("ABORT", "缺数据"))
ok("CONTINUE", main._signal("CONTINUE: 先查站点") == ("CONTINUE", "先查站点"))
ok("不带信号 → 空", main._signal("今天多云，17℃") == ("", ""))
ok("信号必须行首", main._signal("我可以 ASK: 一下吗") == ("", ""), main._signal("我可以 ASK: 一下吗"))
ok("信号前有空格也认", main._signal("   ASK: 问题") == ("ASK", "问题"))
ok("小写也认", main._signal("abort: 算了吧") == ("ABORT", "算了吧"))

# ---------------- 2. _run_tool_loop 的状态 ----------------
check("2. 工具循环状态")
install(Script(executor=[AIMessage(content="ASK: 要哪一天的票？")]))
final, msgs, trace, status = main._run_tool_loop(FakeModel(), [HumanMessage(content="x")])
ok("ASK → status ask", status == "ask", status)
ok("ASK 只调用一次模型", len(msgs) == 2, len(msgs))

install(Script(executor=[AIMessage(content="ABORT: 缺关键数据")]))
_, _, _, status = main._run_tool_loop(FakeModel(), [HumanMessage(content="x")])
ok("ABORT → status abort", status == "abort", status)

install(Script(executor=[AIMessage(content="CONTINUE: 还得再查一次")]))
_, _, _, status = main._run_tool_loop(FakeModel(), [HumanMessage(content="x")])
ok("CONTINUE → status blocked", status == "blocked", status)

install(Script(executor=[AIMessage(content="今天多云，17℃")]))
_, _, trace, status = main._run_tool_loop(FakeModel(), [HumanMessage(content="x")])
ok("正常收尾 → status ok", status == "ok", status)
ok("没调工具时 trace 为空", trace == [], trace)

_call = AIMessage(content="", tool_calls=[{"name": "get_today", "args": {}, "id": "c1"}])
install(Script(executor=[_call, AIMessage(content="今天是 2026-11-20")]))
_, msgs, trace, status = main._run_tool_loop(FakeModel(), [HumanMessage(content="x")])
ok("调工具后正常收尾 → ok", status == "ok", status)
ok("证据链记录了工具名", trace and "get_today" in trace[0], trace)

install(Script(executor=[_call] * 10))
_, _, trace, status = main._run_tool_loop(FakeModel(), [HumanMessage(content="x")], max_iters=3)
ok("轮次耗尽 → blocked", status == "blocked", status)
ok("轮次耗尽仍有证据链", len(trace) == 3, len(trace))

# ---------------- 3. 执行节点的状态与目标联动 ----------------
check("3. 执行节点写回目标状态")
goals.reset()
gid = goals.add_goal("t-react", "查天气", "①查天气", ["给出温度"])
install(Script(executor=[AIMessage(content="ASK: 你想查哪个城市？")]))
state = {"messages": [HumanMessage(content="查天气")], "plan": "①查天气", "goal_id": gid,
         "checks": ["给出温度"], "execution_trace": "", "review": "", "retry_count": 0}
out = main.executor_node(state, {"configurable": {"thread_id": "t-react"}})
ok("ASK 写入 plan_status", out["plan_status"] == "ask", out["plan_status"])
ok("ASK 的用户可见文本带 🤔", out["execution_result"].startswith("🤔"), out["execution_result"])
ok("ASK 目标状态记为 asked", goals.get_goal(gid)["status"] == "asked", goals.get_goal(gid)["status"])

gid2 = goals.add_goal("t-react", "查天气", "①查天气", [])
install(Script(executor=[AIMessage(content="ABORT: 没有该城市的天气源")]))
state["goal_id"] = gid2
out = main.executor_node(state, {"configurable": {"thread_id": "t-react"}})
ok("ABORT 目标状态记为 aborted", goals.get_goal(gid2)["status"] == "aborted",
   goals.get_goal(gid2)["status"])
ok("ABORT 文本可读", "做不到" in out["execution_result"], out["execution_result"])

gid3 = goals.add_goal("t-react", "查天气", "①查天气", [])
install(Script(executor=[AIMessage(content="CONTINUE: 先查站点再查车次")]))
state["goal_id"] = gid3
state["replans"] = 0
out = main.executor_node(state, {"configurable": {"thread_id": "t-react"}})
ok("CONTINUE 累加 replans", out["replans"] == 1, out["replans"])
ok("CONTINUE 记到库里的 replans", goals.get_goal(gid3)["replans"] == 1,
   goals.get_goal(gid3)["replans"])

# ---------------- 4. 整图：重规划回边 ----------------
check("4. 整图：计划被推翻一次")
goals.reset()
main.bind_thread("t-graph")
graph = main.build_graph(None)
script = Script(
    planner=[PLAN_JSON, PLAN_JSON_2],
    executor=[
        AIMessage(content="CONTINUE: 天气源没返回，换个方式再查"),
        AIMessage(content="上海多云，17℃"),
    ],
    critic=[PASS_JSON],
)
install(script)
res = graph.invoke({"messages": [HumanMessage(content="查一下天气")]})
planner_calls = [c for c in script.calls if "规划" in c]
ok("规划节点被调用两次（初始 + 重规划）", len(planner_calls) == 2, script.calls)
ok("最终结果含温度", "17℃" in (res.get("execution_result") or ""), res.get("execution_result"))
ok("state 记录 replans=1", res.get("replans") == 1, res.get("replans"))
all_goals = goals.list_goals(limit=10)
ok("重规划复用同一条目标记录", len(all_goals) == 1, len(all_goals))
ok("目标用的是第二版计划", all_goals[0]["plan"].startswith("①换个城市"), all_goals[0]["plan"])
ok("评审通过时目标置为 done", all_goals[0]["status"] == "done", all_goals[0]["status"])

check("5. 整图：ASK 直接结束，不进评审")
goals.reset()
main.bind_thread("t-graph2")
script = Script(
    planner=[PLAN_JSON],
    executor=[AIMessage(content="ASK: 你想查哪个城市？")],
    critic=[],
)
install(script)
res2 = main.build_graph(None).invoke({"messages": [HumanMessage(content="查天气")]})
ok("没有调用评审节点", not any("评审" in c for c in script.calls), script.calls)
ok("用户看到的是提问", (res2.get("execution_result") or "").startswith("🤔"), res2.get("execution_result"))
ok("目标状态为 asked", goals.list_goals(limit=1)[0]["status"] == "asked",
   goals.list_goals(limit=1)[0]["status"])

check("6. 整图：规划缺参数直接反问，不建执行目标")
goals.reset()
main.bind_thread("t-graph3")
script = Script(planner=['{"plan": "", "checks": [], "need_input": "要盯哪一天的票？"}'],
                executor=[], critic=[])
install(script)
res3 = main.build_graph(None).invoke({"messages": [HumanMessage(content="帮我盯票")]})
ok("没有进执行节点", not any("执行" in c for c in script.calls), script.calls)
ok("反问文本正确", "要盯哪一天的票" in (res3.get("execution_result") or ""),
   res3.get("execution_result"))

print(f"\n结果：{_passed} 通过，{_failed} 失败")
sys.exit(1 if _failed else 0)
