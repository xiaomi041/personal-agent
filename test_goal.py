# -*- coding: utf-8 -*-
"""目标对象化（goals.py）+ 规划/评审解析的离线断言。

跑：python test_goal.py
不联网、不调模型，纯解析与 SQLite 断言。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP_BASE, exist_ok=True)
_tmp = tempfile.mkdtemp(prefix="goal-test-", dir=_TMP_BASE)
os.environ["GOALS_DB"] = str(Path(_tmp) / "goals.db")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import goals  # noqa: E402  必须在设置 GOALS_DB 之后导入
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


# ---------------- 1. _parse_plan ----------------
check("1. 规划输出解析")
plan, checks, need, steps = main._parse_plan(
    '{"plan": "①查文档 ②总结", "checks": ["回答含违约金 5%", "给出了页码"], "need_input": ""}'
)
ok("单行编号计划被拆成两步", [s["text"] for s in steps] == ["查文档", "总结"], steps)
ok("计划渲染成带勾选框的步骤表", plan.startswith("⬜ 1. 查文档"), plan)
ok("正常 JSON：判据 2 条", checks == ["回答含违约金 5%", "给出了页码"], checks)
ok("正常 JSON：无追问", need == "", need)

plan, checks, need, steps = main._parse_plan(
    '{"steps": ["读合同找违约金条款", "把比例算成钱"], "checks": ["给出 5%"], "need_input": ""}'
)
ok("steps 数组：两步", [s["text"] for s in steps] == ["读合同找违约金条款", "把比例算成钱"], steps)
ok("steps 数组：初始状态都是 pending", all(s["status"] == "pending" for s in steps), steps)

plan, checks, need, steps = main._parse_plan(
    '```json\n{"plan": "①查天气", "checks": ["给出温度"], "need_input": ""}\n```'
)
ok("带代码围栏也能解析", len(steps) == 1 and checks == ["给出温度"], (steps, checks))

plan, checks, need, steps = main._parse_plan(
    '好的，我按下面执行：{"plan": "①列待办", "checks": ["列出待办条目"], "need_input": ""}'
)
ok("JSON 前有解释文字也能解析", len(steps) == 1 and plan.startswith("⬜ 1. 列待办"), (plan, steps))

plan, checks, need, steps = main._parse_plan(
    '{"plan": "", "checks": [], "need_input": "要盯哪一天的票？"}'
)
ok("need_input 优先：计划为空", plan == "" and checks == [] and steps == [], (plan, checks, steps))
ok("need_input 取出问题", need == "要盯哪一天的票？", need)

plan, checks, need, steps = main._parse_plan("需要补充：请告诉我出发日期")
ok("老格式「需要补充」仍可用", need == "请告诉我出发日期", need)

plan, checks, need, steps = main._parse_plan("计划：①先查日期 ②再查天气")
ok("纯文本计划兜底（不强拆步骤）",
   plan.startswith("计划：") and checks == [] and need == "" and steps == [], (plan, checks, need, steps))

plan, checks, need, steps = main._parse_plan(
    json.dumps({"plan": "x", "checks": ["a", "b", "c", "d", "e"], "need_input": ""})
)
ok("判据最多保留 3 条", len(checks) == 3, checks)
ok("不编号的字符串不硬拆成步骤", steps == [] and plan == "x", (plan, steps))

ok("空输入不炸", main._parse_plan("")[0] == "")

check("1b. 步骤工具函数")
_st = main._norm_steps(["  1. 读文档  ", "", "②算钱"])
ok("步骤编号连续、清掉前缀", [s["n"] for s in _st] == [1, 2] and _st[1]["text"] == "算钱", _st)
ok("步骤最多 8 条", len(main._norm_steps([f"步骤{i}" for i in range(20)])) == 8, None)
_marks = main._step_marks("先查一下\nSTEP: 2 done\nSTEP: 3 failed: 文件名不对\nSTEP: 4 skipped")
ok("勾选解析出三条", [(n, s) for n, s, _ in _marks] == [(2, "done"), (3, "failed"), (4, "skipped")], _marks)
ok("失败原因被带上", _marks[1][2] == "文件名不对", _marks[1])
ok("句中的 STEP 不算（必须行首）",
   main._step_marks("我把 STEP: 1 done 写在了句子中间") == [], None)
_m2, bad = main._apply_marks(
    [{"n": 1, "text": "a", "status": "pending"}, {"n": 2, "text": "b", "status": "pending"}],
    [(1, "done", ""), (2, "failed", "超时")])
ok("勾选写回状态", [s["status"] for s in _m2] == ["done", "failed"], _m2)
ok("有失败步骤会被识别", bad is True, bad)
ok("越界编号忽略", main._apply_marks([{"n": 1, "text": "a", "status": "pending"}], [(9, "done", "")])[0][0]["status"] == "pending", None)
ok("NEXT 解析 critic", main._next_hint("做完了\nNEXT: critic") == "critic", None)
ok("NEXT 解析 replan", main._next_hint("计划有问题\nnext：replan") == "replan", None)
ok("NEXT 缺失返回空", main._next_hint("做完了，没有声明") == "", None)

# ---------------- 2. _parse_review ----------------
check("2. 评审输出解析")
v, failed, reason = main._parse_review('{"verdict": "pass", "failed": [], "reason": "都对"}')
ok("JSON pass", v == "pass" and failed == [], (v, failed))
v, failed, reason = main._parse_review(
    '{"verdict": "fix", "failed": ["没给页码"], "reason": "引用缺失"}'
)
ok("JSON fix 带未达成判据", v == "fix" and failed == ["没给页码"], (v, failed))
ok("JSON reason 取出", reason == "引用缺失", reason)
v, _, _ = main._parse_review("通过。执行完整。")
ok("老格式「通过」兜底", v == "pass", v)
v, _, _ = main._parse_review("需修正：日期不对")
ok("老格式「需修正」兜底", v == "fix", v)
v, _, reason = main._parse_review("""```json
{"verdict": "fix", "failed": ["x"], "reason": "y"}
```""")
ok("评审也容忍代码围栏", v == "fix" and reason == "y", (v, reason))

# ---------------- 3. goals 存储 ----------------
check("3. 目标存储")
goals.reset()
gid = goals.add_goal("t1", "帮我把合同总结成三段", "①读 ②写", ["回答含违约金", "给出页码"])
g = goals.get_goal(gid)
ok("新建目标可读回", g["id"] == gid and g["thread_id"] == "t1", g)
ok("判据反序列化成 list", g["checks"] == ["回答含违约金", "给出页码"], g["checks"])
ok("初始状态 active", g["status"] == goals.STATUS_ACTIVE, g["status"])

goals.set_status(gid, goals.STATUS_DONE, "已总结")
ok("set_status 生效", goals.get_goal(gid)["status"] == "done")
ok("result 一并写入", goals.get_goal(gid)["result"] == "已总结")

try:
    goals.set_status(gid, "完成丁")
    ok("非法状态被拒", False, "没有抛 KeyError")
except KeyError:
    ok("非法状态被拒", True)

goals.update_plan(gid, "①重读 ②重写", ["新判据"])
ok("update_plan 换计划", goals.get_goal(gid)["plan"] == "①重读 ②重写")
ok("update_plan 换判据", goals.get_goal(gid)["checks"] == ["新判据"])

ok("bump_replan 返回新值", goals.bump_replan(gid) == 1)
ok("bump_replan 累加", goals.bump_replan(gid) == 2)

gid2 = goals.add_goal("t2", "另一个目标", "计划", [])
active = goals.pending(10)
ok("pending 只返回 active", [x["id"] for x in active] == [gid2], [x["id"] for x in active])
ok("list_goals 按线程过滤", len(goals.list_goals(thread_id="t1")) == 1)
ok("list_goals 按状态过滤", len(goals.list_goals(status="done")) == 1)

st = goals.stats(days=1)
ok("stats 统计 total", st["total"] == 2, st)
ok("stats 统计 done", st["done"] == 1, st)
ok("stats 统计 replans", st["replans"] == 2, st)
ok("stats 成功率是比值", abs(st["success_rate"] - 0.5) < 1e-9, st)

# ---------------- 4. 路由函数 ----------------
check("4. 路由")
ok("ask → end", main._after_executor({"plan_status": "ask"}) == "end")
ok("abort → end", main._after_executor({"plan_status": "abort"}) == "end")
ok("blocked 且未超上限 → planner",
   main._after_executor({"plan_status": "blocked", "replans": 0}) == "planner")
ok("blocked 且已用掉重规划 → critic",
   main._after_executor({"plan_status": "blocked", "replans": 1}) == "critic")
ok("未完成阶段默认 → critic", main._after_executor({}) == "critic")
ok("规划 need_input → end", main._plan_router({"plan": "需要补充：x"}) == "end")
ok("规划 ask 状态 → end", main._plan_router({"plan": "x", "plan_status": "ask"}) == "end")
ok("规划正常 → executor", main._plan_router({"plan": "计划：①"}) == "executor")

print(f"\n结果：{_passed} 通过，{_failed} 失败")
sys.exit(1 if _failed else 0)
