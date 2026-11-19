# -*- coding: utf-8 -*-
"""多目标并发推进的对照测量：串行 vs 并行。

做法：造 N 条"停在半路"的目标（互不相同会话），分别用 concurrency=1 与 concurrency=N 跑，
量墙钟、成功率、每条平均耗时。用的是**真实模型**，所以会花钱（3 条目标约 ¥0.1–0.2）。

跑法（容器内）：
    docker compose cp PersonalAI/bench_goals.py personal-ai:/app/bench_goals.py
    docker compose exec -T personal-ai python bench_goals.py
"""
import json
import os
import statistics
import sys
import tempfile
import time

_TMP = tempfile.mkdtemp(prefix="bench-goals-")
os.environ["GOALS_DB"] = os.path.join(_TMP, "goals.db")
os.environ["DATA_DIR"] = _TMP

import goals  # noqa: E402
import main  # noqa: E402
import worker  # noqa: E402

GOAL_TEXT = "现在几点了"
N = int(os.getenv("BENCH_GOALS_N", "3"))


def _age(gid: int, seconds: float = 999) -> None:
    import sqlite3

    con = sqlite3.connect(str(goals.GOALS_DB))
    con.execute("UPDATE goals SET updated_at=? WHERE id=?", (time.time() - seconds, gid))
    con.commit()
    con.close()


def _prepare(n: int) -> list:
    goals.reset()
    ids = []
    for i in range(n):
        gid = goals.add_goal(f"bench-thread-{i}", GOAL_TEXT, "①查时间", ["给出当前时间"])
        _age(gid)
        ids.append(gid)
    return ids


def _run(n: int, concurrency: int, graph) -> dict:
    ids = _prepare(n)
    t0 = time.perf_counter()
    out = worker.resume_goals(limit=n, graph=graph, concurrency=concurrency)
    wall = time.perf_counter() - t0
    stats = goals.stats(1)
    return {
        "concurrency": concurrency,
        "targets": n,
        "wall_s": round(wall, 2),
        "per_goal_s": round(wall / max(1, n), 2),
        "done": out["done"],
        "failed": out["failed"],
        "left": out["left"],
        "success_rate": r"%.0f%%" % (100.0 * out["done"] / max(1, n)),
        "stored": {k: stats[k] for k in ("total", "done", "success_rate")},
        "ids": ids,
    }


def main_() -> int:
    print(f"目标数：{N}（每条都是「{GOAL_TEXT}」）\n")
    graph = main.get_graph()
    rows = []
    for conc in (1, N):
        row = _run(N, conc, graph)
        rows.append(row)
        print(f"  concurrency={conc}: 墙钟 {row['wall_s']}s | 每条 {row['per_goal_s']}s | "
              f"完成 {row['done']}/{row['targets']}")
    serial, parallel = rows[0], rows[-1]
    speedup = serial["wall_s"] / max(0.01, parallel["wall_s"])
    print(f"\n串行 {serial['wall_s']}s → 并行 {parallel['wall_s']}s，"
          f"加速 {speedup:.2f}×（理论上限 {min(N, worker.GOAL_CONCURRENCY)}×）")
    print("\n" + json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
