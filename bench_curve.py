"""任务级成功率的「曲线」：同一套 40 个任务，重复跑 N 轮。

一次 40/40 只是快照——可能是题太简单，也可能是运气好。真正要看的是：
- 每轮成功率是否稳定（波动多大）
- 哪些任务在打摆（同一题有时过有时不过）
- 每轮的成本是否漂移

用法（容器内）：
    python bench_curve.py              # 默认 3 轮
    BENCH_ROUNDS=5 python bench_curve.py
    python bench_curve.py --only 天气   # 只跑名字含关键字的任务

成本提示：一轮 40 个任务约 4 分钟、约 ¥1；3 轮 ≈ 12 分钟。
"""

from __future__ import annotations

import argparse
import os
import statistics
import statistics as st
import sys
import time

import bench_tasks as bt


def run_round(idx: int, cases: list) -> list[dict]:
    """跑一轮：新会话 + 重新备语料，返回每个任务的结果。"""
    bt.THREAD = f"curve-{time.strftime('%m%d-%H%M%S')}-r{idx}"
    bt.prep()
    main = bt.main
    main.bind_thread(bt.THREAD)

    rows = []
    for cat, name, turns, judge in cases:
        t0 = time.perf_counter()
        try:
            r = bt.run_task(turns)
            ok = bool(judge(r["ans"], r["trace"]))
        except Exception as e:  # noqa: BLE001 — 单个任务炸掉不该中断整轮
            r = {"ans": f"<异常：{type(e).__name__}>", "tin": 0, "tout": 0, "calls": 0}
            ok = False
        rows.append({
            "cat": cat, "name": name, "ok": ok,
            "tin": r.get("tin", 0), "tout": r.get("tout", 0),
            "calls": r.get("calls", 0),
            "seconds": time.perf_counter() - t0,
        })
        time.sleep(bt.SLEEP_BETWEEN)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=int(os.getenv("BENCH_ROUNDS", "3")))
    ap.add_argument("--only", default=os.getenv("BENCH_ONLY", ""))
    args = ap.parse_args()

    cases = [t for t in bt.TASKS if args.only in t[1]] if args.only else list(bt.TASKS)
    print("=" * 72)
    print(f"任务级成功率曲线：{len(cases)} 个任务 × {args.rounds} 轮")
    print("=" * 72)

    rounds = []
    for i in range(args.rounds):
        t0 = time.time()
        rows = run_round(i, cases)
        passed = sum(1 for r in rows if r["ok"])
        tin = sum(r["tin"] for r in rows)
        rounds.append(rows)
        print(f"  第 {i + 1} 轮：{passed}/{len(rows)} "
              f"（{passed / len(rows):.1%}）| {tin} 输入 token | {time.time() - t0:.0f}s")

    # ---- 每轮成功率与波动
    rates = [sum(1 for r in rs if r["ok"]) / len(rs) for rs in rounds]
    print()
    print("一、每轮成功率")
    for i, r in enumerate(rates):
        bar = "█" * int(round(r * 40))
        print(f"  第 {i + 1} 轮  {r:6.1%}  {bar}")
    print(f"  平均 {statistics.mean(rates):.1%}，极差 {max(rates) - min(rates):.1%}，"
          f"标准差 {statistics.pstdev(rates):.3f}")

    # ---- 稳定性：每个任务过了几轮
    names = [t[1] for t in cases]
    per = {n: [rs[i]["ok"] for rs in rounds] for i, n in enumerate(names)}
    stable_pass = [n for n, v in per.items() if all(v)]
    stable_fail = [n for n, v in per.items() if not any(v)]
    flaky = [n for n, v in per.items() if any(v) and not all(v)]

    print()
    print("二、稳定性")
    print(f"  每次都过  {len(stable_pass)} 个")
    print(f"  每次都挂  {len(stable_fail)} 个" + (f"：{'、'.join(stable_fail)}" if stable_fail else ""))
    print(f"  时过时挂  {len(flaky)} 个" + (f"：{'、'.join(flaky)}" if flaky else ""))
    if flaky:
        print("  （时过时挂的任务说明判定或模型行为不稳，单独看这三条比看总成功率有用）")
        for n in flaky:
            print(f"    - {n}: {' '.join('✅' if x else '❌' for x in per[n])}")

    # ---- 成本漂移
    print()
    print("三、成本漂移")
    for i, rs in enumerate(rounds):
        tin = sum(r["tin"] for r in rs)
        secs = sum(r["seconds"] for r in rs)
        print(f"  第 {i + 1} 轮  {tin:>7} 输入 token | {secs:5.0f}s | "
              f"平均 {st.mean([r['calls'] for r in rs]):.1f} 次调用/任务")

    return 0 if all(r >= 0.99 for r in rates) else 1


if __name__ == "__main__":
    sys.exit(main())
