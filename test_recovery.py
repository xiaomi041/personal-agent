# -*- coding: utf-8 -*-
"""目标失败后换策略（recovery）的离线断言。

跑：python test_recovery.py
不联网、不调模型：策略选择是纯函数，worker 用假 graph。
"""
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_TMP_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP_BASE, exist_ok=True)
_tmp = tempfile.mkdtemp(prefix="recovery-test-", dir=_TMP_BASE)
os.environ["GOALS_DB"] = str(Path(_tmp) / "goals.db")
os.environ["DATA_DIR"] = _tmp
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import goals  # noqa: E402
import recovery  # noqa: E402
import worker  # noqa: E402

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


class FakeGraph:
    """跑一次就把目标置为预设状态，并记录 state。"""

    def __init__(self, finish="done", boom=False):
        self.finish = finish
        self.boom = boom
        self.states = []

    def invoke(self, state, config=None):
        self.states.append(state)
        if self.boom:
            raise RuntimeError("模型挂了")
        gid = state.get("goal_id")
        if gid and self.finish == "done":
            goals.set_status(gid, goals.STATUS_DONE, "做完了")
        elif gid and self.finish == "active":
            goals.set_status(gid, goals.STATUS_ACTIVE)
            goals.note_error(gid, state.get("last_error") or "还是不行")
        return {"execution_result": "ok"}


def age(goal_id, seconds=999):
    con = sqlite3.connect(str(goals.GOALS_DB))
    con.execute("UPDATE goals SET updated_at=? WHERE id=?", (time.time() - seconds, goal_id))
    con.commit()
    con.close()


# ---------------- 1. 策略选择表 ----------------
check("1. 按失败原因挑策略")
r = recovery.pick("调用 write_file 时报错：目录不存在")
ok("点名工具 → switch_tool", r.strategy == "switch_tool", r)
ok("提示里给出替代做法", "outputs" in r.hint, r.hint)

r = recovery.pick("读取失败：文件不存在")
ok("缺信息 → ask_user", r.strategy == "ask_user", r)

r = recovery.pick("Permission denied: 无法写入该目录")
ok("权限问题 → narrow", r.strategy == "narrow", r)

r = recovery.pick("模型输出格式不对")
ok("无线索 → replan", r.strategy == "replan", r)
ok("replan 提示要求换做法", "完全不同" in r.hint, r.hint)

# 已经 replan 过：不能再给 replan
r2 = recovery.pick("模型输出格式不对", tried=["replan"])
ok("replan 试过就换别的", r2.strategy in ("switch_tool", "narrow"), r2)

# 次数用尽
r3 = recovery.pick("随便什么原因", tried=["replan", "narrow"], max_recoveries=2)
ok("次数用尽 → giveup", r3.strategy == "giveup", r3)
ok("giveup 带理由", "2 次" in r3.reason, r3.reason)
ok("giveup 不算换策略", r3.giveup is True)

# ask_user 也试过了就不该再问
r4 = recovery.pick("文件不存在", tried=["ask_user"])
ok("问过用户就不再问", r4.strategy != "ask_user", r4)

# ---------------- 2. 给规划节点的提示 ----------------
check("2. 提示文案")
rec = recovery.Recovery("replan", "换一条路线", "上次超时")
hint = recovery.hint_for_planner(rec, "上次超时")
ok("含失败原因", "上次超时" in hint, hint)
ok("含换策略指令", "换一条路线" in hint, hint)
ok("giveup 时不催它换", "必须换" not in recovery.hint_for_planner(
    recovery.Recovery("giveup", "", ""), "x"))
ok("summarize 是中文", recovery.summarize(["replan", "narrow"]) == "换计划、缩小范围",
   recovery.summarize(["replan", "narrow"]))
ok("retry_state 带 hint", recovery.retry_state(rec)["retry_hint"] == "换一条路线")

# ---------------- 3. 续跑次数用尽 → 换策略再跑一次 ----------------
check("3. 次数用尽后换策略并立刻重试")
goals.reset()
g1 = goals.add_goal("t-rec-1", "把不存在的文件总结一下", "①读文件", [])
goals.note_error(g1, "文件不存在")
for _ in range(worker.GOAL_MAX_ATTEMPTS):
    goals.bump_attempt(g1)
age(g1)
graph = FakeGraph(finish="done")
out = worker.resume_goals(graph=graph)
ok("这一次算成功", out["done"] == 1, out)
ok("graph 被调用了一次", len(graph.states) == 1, len(graph.states))
ok("state 带上换策略提示", "缺的具体信息" in (graph.states[0].get("retry_hint") or ""),
   graph.states[0].get("retry_hint"))
ok("state 带上失败原因", "文件不存在" in (graph.states[0].get("last_error") or ""),
   graph.states[0].get("last_error"))
data = goals.get_goal(g1)
ok("换策略被记录", [r["strategy"] for r in data["recovery"]] == ["ask_user"], data["recovery"])
ok("续跑计数被重置", data["attempts"] == 0, data["attempts"])
ok("目标状态 done", data["status"] == goals.STATUS_DONE, data["status"])

# ---------------- 4. 换策略也失败 → 用尽后放弃 ----------------
check("4. 策略用尽才放弃，且写清试过什么")
goals.reset()
g2 = goals.add_goal("t-rec-2", "注定做不成的活", "①", [])
goals.note_error(g2, "随便什么原因")
# 已有两次换策略记录 + 续跑次数再次用尽
goals.add_recovery(g2, "replan", "")
goals.add_recovery(g2, "narrow", "")
for _ in range(worker.GOAL_MAX_ATTEMPTS):
    goals.bump_attempt(g2)
age(g2)
graph = FakeGraph(finish="done")
out = worker.resume_goals(graph=graph)
ok("记为失败", out["failed"] == 1, out)
ok("graph 没被调用", graph.states == [], graph.states)
data = goals.get_goal(g2)
ok("状态 aborted", data["status"] == goals.STATUS_ABORTED, data["status"])
ok("结果里有「换策略仍没做成」", "换策略仍没做成" in (data["result"] or ""), data["result"])
ok("结果里列出试过的策略", "换计划" in (data["result"] or ""), data["result"])

# ---------------- 5. 调用抛异常也要记下原因，供下次归因 ----------------
check("5. 异常也要留下失败原因")
goals.reset()
g3 = goals.add_goal("t-rec-3", "会抛异常的活", "①", [])
age(g3)
out = worker.resume_goals(graph=FakeGraph(boom=True))
ok("记为失败", out["failed"] == 1, out)
data = goals.get_goal(g3)
ok("last_error 非空", "模型挂了" in (data["last_error"] or ""), data["last_error"])
ok("目标仍 active（留给换策略）", data["status"] == goals.STATUS_ACTIVE, data["status"])

# ---------------- 6. 每换一次策略都重新给续跑次数 ----------------
check("6. 换策略后 attempts 重置，不会立刻又判死")
goals.reset()
g4 = goals.add_goal("t-rec-4", "权限被拒绝的活", "①", [])
goals.note_error(g4, "Permission denied")
for _ in range(worker.GOAL_MAX_ATTEMPTS):
    goals.bump_attempt(g4)
age(g4)
graph = FakeGraph(finish="active")  # 换策略后仍没成
out = worker.resume_goals(graph=graph)
ok("这次不算 done", out["done"] == 0, out)
data = goals.get_goal(g4)
ok("记了一条 narrow 策略", [r["strategy"] for r in data["recovery"]] == ["narrow"], data["recovery"])
ok("attempts 被重置后又被本次计 0", data["attempts"] == 0, data["attempts"])
ok("目标回到 active 等待下次", data["status"] == goals.STATUS_ACTIVE, data["status"])

print(f"\n结果：{_passed} 通过，{_failed} 失败")
sys.exit(1 if _failed else 0)
