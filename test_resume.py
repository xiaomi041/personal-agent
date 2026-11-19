# -*- coding: utf-8 -*-
"""目标队列续跑（worker.resume_goals）的离线断言。

跑：python test_resume.py
用假 graph，不联网、不调模型。
"""
import os
import sys
import tempfile
import threading
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

    def __init__(self, finish="done", boom=False, delay=0.0):
        self.finish = finish
        self.boom = boom
        self.delay = delay
        self.calls = []
        self.states = []
        self.spans = []
        self._lock = threading.Lock()

    def invoke(self, state, config=None):
        text = state["messages"][0].content
        thread = (config or {}).get("configurable", {}).get("thread_id")
        started = time.perf_counter()
        with self._lock:
            self.calls.append((text, thread, state.get("resume")))
            self.states.append(state)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.boom:
                raise RuntimeError("模型挂了")
            # 用 state 里带的 goal_id 更接近真实行为（规划节点就是靠它复用记录）
            gid = state.get("goal_id") or goals.list_goals(thread_id=thread, limit=1)[0]["id"]
            if self.finish == "done":
                goals.set_status(gid, goals.STATUS_DONE, "续跑完成")
            return {"execution_result": "ok"}
        finally:
            with self._lock:
                self.spans.append((thread, started, time.perf_counter()))



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

# ---------------- 4. 续跑次数用尽 → 先换策略，不是判死 ----------------
check("4. 续跑超限后换策略重试（不再直接判失败）")
goals.reset()
g3 = goals.add_goal("t-c", "总是跑不完的目标", "①", [])
goals.note_error(g3, "模型输出格式不对")
age(g3, 999)
for _ in range(worker.GOAL_MAX_ATTEMPTS):
    goals.bump_attempt(g3)
age(g3, 999)  # bump_attempt 会刷新 updated_at，重新拨旧
graph = FakeGraph(finish="done")
out = worker.resume_goals(graph=graph)
ok("换策略后这一轮成了", out["done"] == 1, out)
ok("graph 被调用一次（换策略后重试）", len(graph.calls) == 1, graph.calls)
ok("带上换策略提示", "换" in (graph.states[0].get("retry_hint") or ""),
   graph.states[0].get("retry_hint"))
ok("换策略被记录", len(goals.get_goal(g3)["recovery"]) == 1, goals.get_goal(g3)["recovery"])

check("4b. 策略用尽才判失败")
goals.reset()
g3b = goals.add_goal("t-c2", "怎么都做不成的目标", "①", [])
goals.note_error(g3b, "随便什么原因")
for s in ("replan", "narrow"):
    goals.add_recovery(g3b, s, "")
for _ in range(worker.GOAL_MAX_ATTEMPTS):
    goals.bump_attempt(g3b)
age(g3b, 999)
graph = FakeGraph(finish="done")
out = worker.resume_goals(graph=graph)
ok("本次记为失败", out["failed"] == 1, out)
ok("graph 没有被调用", graph.calls == [], graph.calls)
ok("目标状态 aborted", goals.get_goal(g3b)["status"] == goals.STATUS_ABORTED,
   goals.get_goal(g3b)["status"])
ok("放弃原因写进 result", "换策略仍没做成" in (goals.get_goal(g3b)["result"] or ""),
   goals.get_goal(g3b)["result"])

# ---------------- 5. 没有目标时不折腾 ----------------
check("5. 空队列")
goals.reset()
out = worker.resume_goals(graph=FakeGraph())
ok("空队列 picked=0", out["picked"] == 0 and out["done"] == 0 and out["failed"] == 0
   and out["left"] == 0, out)


# ---------------- 6. 完成要写提醒弹窗 ----------------
check("6. 目标完成写一条提醒（用户不必盯着页面）")
import scheduler as _sched  # noqa: E402

goals.reset()
alerts = []
_orig_push = _sched.push_alert
_sched.push_alert = lambda content: alerts.append(content) or 1
try:
    g4 = goals.add_goal("t-d", "查明天北京天气", "①查天气", ["给出温度"])
    age(g4, 999)
    out = worker.resume_goals(graph=FakeGraph(finish="done"))
    ok("续跑成功", out["done"] == 1, out)
    ok("写了一条提醒", len(alerts) == 1, alerts)
    ok("提醒里有「目标完成」", alerts and "目标完成" in alerts[0], alerts)
    ok("提醒里带原委托", alerts and "查明天北京天气" in alerts[0], alerts)

    # 通知本身失败不能影响续跑结果
    def _boom(_content):
        raise RuntimeError("弹窗库挂了")

    _sched.push_alert = _boom
    goals.reset()
    g5 = goals.add_goal("t-e", "再来一次", "①", [])
    age(g5, 999)
    out = worker.resume_goals(graph=FakeGraph(finish="done"))
    ok("通知抛异常仍算完成", out["done"] == 1, out)
    ok("目标仍是 done", goals.get_goal(g5)["status"] == goals.STATUS_DONE)
finally:
    _sched.push_alert = _orig_push

# ---------------- 7. 同时推进多个目标 ----------------
check("7. 多目标并发推进")
goals.reset()
ids = []
for i in range(3):
    gid = goals.add_goal(f"t-par-{i}", f"任务{i}", "①", [])
    age(gid, 999)
    ids.append(gid)
graph = FakeGraph(finish="done", delay=0.35)
t0 = time.perf_counter()
out = worker.resume_goals(limit=10, graph=graph, concurrency=3)
elapsed = time.perf_counter() - t0
ok("三条都完成", out["done"] == 3, out)
ok("识别出 3 个会话", out["groups"] == 3, out)
# 串行至少 1.05s，并行应接近 0.35s；留足余量避免 CI 抖动
ok("并行确实更快（< 0.8s）", elapsed < 0.8, f"{elapsed:.3f}s")
ok("三条各调一次", len(graph.calls) == 3, len(graph.calls))

check("7b. 同一会话必须串行（共用 checkpoint）")
goals.reset()
same = [goals.add_goal("t-same", f"同会话任务{i}", "①", []) for i in range(2)]
for gid in same:
    age(gid, 999)
graph = FakeGraph(finish="done", delay=0.25)
t0 = time.perf_counter()
out = worker.resume_goals(limit=10, graph=graph, concurrency=3)
elapsed = time.perf_counter() - t0
ok("同会话两条都完成", out["done"] == 2, out)
ok("只算一个会话组", out["groups"] == 1, out)
ok("串行耗时 ≥ 0.45s", elapsed >= 0.45, f"{elapsed:.3f}s")
spans = sorted(graph.spans, key=lambda s: s[1])
ok("两段执行不重叠", len(spans) == 2 and spans[1][1] >= spans[0][2] - 0.02, spans)

check("7c. 并发度可被压到 1（退化成串行）")
goals.reset()
for i in range(3):
    age(goals.add_goal(f"t-serial-{i}", f"串行{i}", "①", []), 999)
graph = FakeGraph(finish="done", delay=0.2)
t0 = time.perf_counter()
out = worker.resume_goals(limit=10, graph=graph, concurrency=1)
elapsed = time.perf_counter() - t0
ok("三条都完成", out["done"] == 3, out)
ok("concurrency=1 时 ≥ 0.55s", elapsed >= 0.55, f"{elapsed:.3f}s")

check("7d. 认领互斥：查到 ≠ 抢到")
goals.reset()
gid = goals.add_goal("t-claim", "会被抢的目标", "①", [])
age(gid, 999)
ok("第一次认领成功", goals.claim(gid, 60) is True)
ok("被别人认领后 stale 不再返回它",
   all(g["id"] != gid for g in goals.stale(120, limit=10)))
ok("第二次认领失败", goals.claim(gid, 60) is False)
goals.release(gid)
ok("释放后又能被认领", goals.claim(gid, 60) is True)
goals.release(gid)

check("7e. 已被认领的目标在续跑里只记 skip，不会重复跑")
goals.reset()
gid = goals.add_goal("t-skip", "别人正在跑的目标", "①", [])
age(gid, 999)
goals.claim(gid, 120)
graph = FakeGraph(finish="done")
out = worker.resume_goals(limit=10, graph=graph)
ok("picked 里没有它", out["picked"] == 0, out)
ok("graph 没被调用", graph.calls == [], graph.calls)

print(f"\n结果：{_passed} 通过，{_failed} 失败")
sys.exit(1 if _failed else 0)
