# -*- coding: utf-8 -*-
"""目标队列续跑（worker.resume_goals）的离线断言。

跑：python test_resume.py
用假 graph，不联网、不调模型。
"""
import os
import sys
import tempfile
import time
from pathlib import Path

_TMP_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP_BASE, exist_ok=True)
_tmp = tempfile.mkdtemp(prefix="resume-test-", dir=_TMP_BASE)
os.environ["GOALS_DB"] = str(Path(_tmp) / "goals.db")
os.environ["DATA_DIR"] = _tmp
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import goals  # noqa: E402
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
    """记录被调用时的 (text, thread)，并按预设决定目标最终状态。"""

    def __init__(self, finish="done", boom=False):
        self.finish = finish
        self.boom = boom
        self.calls = []
        self.states = []

    def invoke(self, state, config=None):
        text = state["messages"][0].content
        thread = (config or {}).get("configurable", {}).get("thread_id")
        self.calls.append((text, thread, state.get("resume")))
        self.states.append(state)
        if self.boom:
            raise RuntimeError("模型挂了")
        # 用 state 里带的 goal_id 更接近真实行为（规划节点就是靠它复用记录）
        gid = state.get("goal_id") or goals.list_goals(thread_id=thread, limit=1)[0]["id"]
        if self.finish == "done":
            goals.set_status(gid, goals.STATUS_DONE, "续跑完成")
        return {"execution_result": "ok"}


def age(goal_id, seconds):
    """把 updated_at 往前拨，模拟「停在半路很久了」。"""
    import sqlite3

    con = sqlite3.connect(str(goals.GOALS_DB))
    con.execute("UPDATE goals SET updated_at=? WHERE id=?", (time.time() - seconds, goal_id))
    con.commit()
    con.close()


# ---------------- 1. stale 的判定 ----------------
check("1. stale 只捞「停在半路」的目标")
goals.reset()
old = goals.add_goal("t-old", "把合同总结成三段", "①读 ②写", ["给出三段"])
fresh = goals.add_goal("t-new", "刚建的", "①", [])
age(old, 999)
got = goals.stale(120, limit=10)
ok("老的被捞出", [g["id"] for g in got] == [old], [g["id"] for g in got])
ok("刚建的不被捞", all(g["id"] != fresh for g in got), got)

done_id = goals.add_goal("t-done", "已完成的目标", "①", [])
goals.set_status(done_id, goals.STATUS_DONE)
age(done_id, 999)
ok("已完成的不被捞", all(g["id"] != done_id for g in goals.stale(120, 10)))

# ---------------- 2. 续跑成功 ----------------
check("2. 续跑并把目标推到 done")
goals.reset()
g1 = goals.add_goal("t-a", "查明天北京天气", "①查天气", ["给出温度"])
age(g1, 999)
graph = FakeGraph(finish="done")
out = worker.resume_goals(graph=graph)
ok("捞到 1 条", out["picked"] == 1, out)
ok("续跑成功 1 条", out["done"] == 1, out)
ok("graph 被调用一次", len(graph.calls) == 1, graph.calls)
ok("续跑带上原委托文本", graph.calls[0][0] == "查明天北京天气", graph.calls)
ok("续跑带原 thread", graph.calls[0][1] == "t-a", graph.calls)
ok("续跑标记 resume=True", graph.calls[0][2] is True, graph.calls)
# 回归：续跑必须带 goal_id 回写同一条记录，否则规划节点会新建目标、原目标永远 active
ok("续跑带上原 goal_id", graph.states[0].get("goal_id") == g1, graph.states[0].get("goal_id"))
ok("续跑带上原判据", graph.states[0].get("checks") == ["给出温度"], graph.states[0].get("checks"))
ok("目标状态变 done", goals.get_goal(g1)["status"] == "done")
ok("attempts 记 1 次", goals.get_goal(g1)["attempts"] == 1, goals.get_goal(g1)["attempts"])

# ---------------- 3. 续跑失败 ----------------
check("3. 续跑抛异常只记失败，不拖垮 worker")
goals.reset()
g2 = goals.add_goal("t-b", "查天气", "①查", [])
age(g2, 999)
out = worker.resume_goals(graph=FakeGraph(boom=True))
ok("记为失败 1 条", out["failed"] == 1, out)
ok("目标仍是 active（留给下次）", goals.get_goal(g2)["status"] == goals.STATUS_ACTIVE,
   goals.get_goal(g2)["status"])
ok("attempts 仍记了数", goals.get_goal(g2)["attempts"] == 1, goals.get_goal(g2)["attempts"])

# ---------------- 4. 续跑次数封顶 ----------------
check("4. 续跑超限就判失败，不再调 graph")
goals.reset()
g3 = goals.add_goal("t-c", "总是跑不完的目标", "①", [])
age(g3, 999)
for _ in range(worker.GOAL_MAX_ATTEMPTS):
    goals.bump_attempt(g3)
age(g3, 999)  # bump_attempt 会刷新 updated_at，重新拨旧
graph = FakeGraph(finish="done")
out = worker.resume_goals(graph=graph)
ok("本次记为失败", out["failed"] == 1, out)
ok("graph 没有被调用", graph.calls == [], graph.calls)
ok("目标状态 aborted", goals.get_goal(g3)["status"] == goals.STATUS_ABORTED,
   goals.get_goal(g3)["status"])
ok("放弃原因写进 result", "续跑" in (goals.get_goal(g3)["result"] or ""),
   goals.get_goal(g3)["result"])

# ---------------- 5. 没有目标时不折腾 ----------------
check("5. 空队列")
goals.reset()
out = worker.resume_goals(graph=FakeGraph())
ok("空队列 picked=0", out == {"picked": 0, "done": 0, "failed": 0, "left": 0}, out)

print(f"\n结果：{_passed} 通过，{_failed} 失败")
sys.exit(1 if _failed else 0)
