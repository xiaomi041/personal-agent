"""量化基准：把"感觉快/感觉好用"变成可复现的数字。

在容器内跑：docker compose exec -T personal-ai python bench.py
不调用真实委托的部分（解析/检索）可离线反复跑；端到端与并发部分会消耗少量 API 额度。

输出四张表：
  1. 解析：各类文档的块数、切片数、字符数、耗时
  2. 检索：N 次查询的 P50/P95 耗时与 top-1 相似度
  3. 端到端：每个任务的模型调用次数、token、耗时
  4. 并发：K 路并发下的整体 P95 与是否串会话
"""
from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import main
import documents as docengine
from langchain_core.messages import HumanMessage

BENCH_THREAD = os.getenv("BENCH_THREAD", "bench-thread")
OUT = []


def say(line: str = "") -> None:
    print(line, flush=True)
    OUT.append(line)


def pct(values, p: float) -> float:
    """简单分位数（nearest-rank），够用且不需要 numpy 依赖。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def median(values) -> float:
    return statistics.median(values) if values else 0.0


# ---------------------------------------------------------------- 1. 解析
def bench_parse() -> None:
    import eval_docs

    samples = [
        ("合同 PDF（2 页，含表格）", eval_docs.build_contract_pdf, "bench-contract.pdf"),
        ("产品手册 DOCX（含表格）", eval_docs.build_manual_docx, "bench-manual.docx"),
        ("长文档 PDF（12 页）", lambda: eval_docs.build_long_pdf(12), "bench-long.pdf"),
        ("扫描件 PNG（OCR）", eval_docs.build_scan_png, "bench-scan.png"),
    ]
    say("## 1. 文档解析")
    say("| 样例 | 块数 | 切片数 | 正文字符 | 耗时(ms) | 每千字耗时(ms) |")
    say("|---|---:|---:|---:|---:|---:|")
    for name, builder, fname in samples:
        data = builder()
        t0 = time.time()
        blocks = docengine.extract_blocks(fname, data)
        t1 = time.time()
        chunks = docengine.chunk_blocks(blocks)
        t2 = time.time()
        chars = sum(len(c.text) for c in chunks)
        cost = (t2 - t0) * 1000
        say(f"| {name} | {len(blocks)} | {len(chunks)} | {chars} | {cost:.0f} | "
            f"{cost / max(chars, 1) * 1000:.2f} |")
    say()


# ---------------------------------------------------------------- 2. 检索
QUERIES = [
    "违约金比例是多少", "合同编号是什么", "签约日期", "保证金多少",
    "争议怎么解决", "蓝鲸计划什么时候上线", "项目负责人是谁",
    "退款政策", "保修期多久", "发票怎么开",
]


def bench_retrieval(rounds: int = 3) -> None:
    say("## 2. 检索延迟（混合召回 + 阈值过滤）")
    lat, sims, hits = [], [], []
    for _ in range(rounds):
        for q in QUERIES:
            t0 = time.time()
            got = main.search_documents(q, thread_id=BENCH_THREAD)
            lat.append((time.time() - t0) * 1000)
            hits.append(1 if got else 0)
            if got:
                sims.append(float(got[0].get("score") or 0))
    say(f"- 查询次数：{len(lat)}（{len(QUERIES)} 条 query × {rounds} 轮）")
    say(f"- P50 {median(lat):.0f} ms / P95 {pct(lat, 95):.0f} ms / 最大 {max(lat):.0f} ms")
    say(f"- 命中率（有结果返回）{sum(hits)}/{len(hits)}")
    if sims:
        say(f"- top-1 相似度：均值 {statistics.mean(sims):.3f}，最低 {min(sims):.3f}")
    say()


# ---------------------------------------------------------------- 3. 端到端
TASKS = [
    ("现在几点了", "单工具、无检索", ""),
    ("帮我加一个待办：明天上午十点交论文初稿", "写操作", ""),
    ("明天十堰天气怎么样", "外部 API", ""),
    ("看看我有哪些待办", "读存储", ""),
    # 文档类任务必须落在语料所在会话，否则规划节点看不到文档
    ("合同里的违约金比例是多少", "文档检索", BENCH_THREAD),
    ("总结一下我上传的文档", "map-reduce 摘要", BENCH_THREAD),
]


def _run_turn(task: str, thread: str) -> dict:
    before = main.usage_snapshot()
    t0 = time.time()
    graph = main.get_graph()
    try:
        res = graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": thread}},
        )
        answer = (res.get("execution_result") or "")[:60]
        err = ""
    except Exception as e:  # noqa: BLE001
        answer, err = "", f"{type(e).__name__}: {e}"
    cost = time.time() - t0
    after = main.usage_snapshot()
    return {
        "task": task,
        "seconds": cost,
        "calls": after["calls"] - before["calls"],
        "in": after["in"] - before["in"],
        "out": after["out"] - before["out"],
        "tools": after["tool_calls"] - before["tool_calls"],
        "answer": answer,
        "err": err,
    }


def bench_end2end() -> list:
    say("## 3. 端到端（每个任务一次真实委托）")
    say("| 任务 | 类型 | 模型调用 | 工具调用 | 输入 token | 输出 token | 耗时(s) |")
    say("|---|---|---:|---:|---:|---:|---:|")
    rows = []
    for task, kind, thread in TASKS:
        r = _run_turn(task, thread or f"{BENCH_THREAD}-{abs(hash(task)) % 1000}")
        rows.append(r)
        flag = f" ⚠{r['err']}" if r["err"] else ""
        say(f"| {task} | {kind} | {r['calls']} | {r['tools']} | {r['in']} | {r['out']} | "
            f"{r['seconds']:.1f}{flag} |")
    total_in = sum(r["in"] for r in rows)
    total_out = sum(r["out"] for r in rows)
    total_s = sum(r["seconds"] for r in rows)
    secs = [r["seconds"] for r in rows]
    say()
    say(f"- 合计 {len(rows)} 次委托：{total_in} 输入 / {total_out} 输出 token，{total_s:.1f} s")
    say(f"- 单次 P50 {median(secs):.1f}s / P95 {pct(secs, 95):.1f}s / 最大 {max(secs):.1f}s")
    say(f"- 单次平均 {total_in / len(rows):.0f} 输入 token，{total_out / len(rows):.0f} 输出 token")
    say()
    return rows


# ---------------------------------------------------------------- 4. 并发
def bench_concurrency(workers: int = 4, per_worker: int = 1) -> None:
    say(f"## 4. 并发 {workers} 路（同时发起，验证不串会话）")

    def one(i: int) -> dict:
        tid = f"{BENCH_THREAD}-c{i}"
        return _run_turn("现在几点了", tid)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, range(workers * per_worker)))
    wall = time.time() - t0
    ok = [r for r in results if not r["err"]]
    secs = [r["seconds"] for r in ok]
    say(f"- 总墙钟 {wall:.1f}s，成功 {len(ok)}/{len(results)}")
    if secs:
        say(f"- 单请求 P50 {median(secs):.1f}s / P95 {pct(secs, 95):.1f}s")
    say(f"- 吞吐 {len(ok) / wall:.2f} 请求/秒")
    say()


def main_() -> None:
    main.add_document  # 触发导入
    import eval_docs  # 复用样例构造器

    # 语料入库（后续检索与问答都依赖它）
    for fname, builder in (
        ("bench-合同.pdf", eval_docs.build_contract_pdf),
        ("bench-手册.docx", eval_docs.build_manual_docx),
        ("bench-长文档.pdf", lambda: eval_docs.build_long_pdf(12)),
        ("bench-扫描件.png", eval_docs.build_scan_png),
    ):
        t0 = time.time()
        msg = main.add_document(fname, builder(), thread_id=BENCH_THREAD)
        say(f"[入库] {fname} {(time.time() - t0) * 1000:.0f} ms → {str(msg)[:60]}")
    say()

    bench_parse()
    bench_retrieval()
    rows = bench_end2end()
    bench_concurrency()

    with open("bench_result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "log": OUT}, f, ensure_ascii=False, indent=2)
    say("结果已写入 bench_result.json")


if __name__ == "__main__":
    main_()
