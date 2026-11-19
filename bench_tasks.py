# -*- coding: utf-8 -*-
"""任务级评测：不问「代码有没有坏」，只问「委托能不能办成」。

与 eval_docs.py 的区别：
- eval_docs 评的是组件（检索命中率、引用准确率、拒答），走的是工具函数；
- 这里评的是**任务成功率**：一整句真实委托丢给状态图，看最终交付对不对，
  同时记录这条任务花了多少次模型调用、多少次工具调用、多少 token、多长时间。

产出四张表：总体 / 分类 / 成本 / 失败清单。

运行：python bench_tasks.py
"""
import io
import os
import re
import statistics
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from langchain_core.messages import HumanMessage

import main

# 每次跑用一个全新会话，否则上一轮的历史会留在 checkpoint 里，
# 让「第一个任务」就背着几十 k 的上下文——测出来的不是产品，是上一次的自己。
THREAD = os.getenv("BENCH_THREAD") or f"bench-task-{time.strftime('%m%d-%H%M%S')}"
DOCS = ("合作协议.pdf", "产品手册.docx", "合同扫描件.png", "长文档.pdf")


# ------------------------------------------------------------------ 判定工具

def has(text: str, *words) -> bool:
    return any(w in text for w in words)


def has_amount(text: str, value: int) -> bool:
    """金额判定：容忍 10000 / 10,000 / 1 万元 / 一万元 / 两万 等写法。

    这一版是第三稿。前两稿都被真实输出打了脸：模型写「1 万元」，
    第一稿找 "1 万"（原文是「1 万 元」，中间有空格）漏判；
    第二稿找「一万」，模型写的是阿拉伯数字「1」。判分函数比模型还脆，就不是好评测。
    """
    t = re.sub(r"\s+", "", text.replace(",", "").replace("，", ""))
    if str(value) in t:
        return True
    for m in re.finditer(r"(\d+(?:\.\d+)?)万", t):
        if abs(float(m.group(1)) * 10000 - value) < 1:
            return True
    zh = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    for m in re.finditer(r"([一两二三四五六七八九十]+)万", t):
        n = zh.get(m.group(1)) if len(m.group(1)) == 1 else None
        if n is not None and n * 10000 == value:
            return True
    return False


def refused(text: str) -> bool:
    """拒答判定：模型换一种说法（「均未记载」「未出现」）也算拒答。

    上一版只列了 9 个说法，模型回「均未记载……未出现……」，判定函数没认出来，
    把一个正确的拒答记成失败——这是判定函数的锅，不是产品的锅。
    """
    return has(text, "没有", "未找到", "无相关", "无法", "不存在", "不清楚", "不包含",
               "未提及", "没有提到", "未记载", "未出现", "查无", "并无", "均无", "均未")


def asked_back(text: str) -> bool:
    """只有「没干活、先要信息」才算澄清追问。

    任务已交付后追一句「需要我帮你新增一条吗？」是礼貌收尾，不是澄清——
    上一版用「含问号」判定，把这种收尾误判成反问（假阴性）。

    第三版：拿「第一句是不是问句」判断，而不是整段。模型这次回的是
    「🤔 提醒的内容和时间分别是什么？（例如：明天 9 点提醒我交实验报告）」，
    整段里有数字（9 点）就被旧规则判成"已交付"，实际它是真追问。
    """
    t = (text or "").strip()
    if t.startswith("🤔") or t.startswith("需要补充"):
        return True
    if not has(t, "？", "?"):
        return False
    first = re.split(r"[。\n]", t)[0]
    return has(first, "？", "?") and not re.search(r"\d", first)


# ------------------------------------------------------------------ 任务定义
# 每项：(类别, 名称, [多轮输入], 判定函数)
# 判定函数拿到 (最终回答, 完整 trace 文本)，返回 True/False

TASKS = [
    # ---- 文档·事实（6）
    ("文档·事实", "违约金比例", ["合作协议里的违约金比例是多少"],
     lambda a, t: has(a, "5%", "百分之五")),
    ("文档·事实", "保证金金额", ["保证金要交多少钱"],
     lambda a, t: has_amount(a, 10000)),
    ("文档·事实", "争议解决方式", ["双方出现争议怎么解决"],
     lambda a, t: has(a, "仲裁")),
    ("文档·事实", "保密期限", ["保密期限是几年"],
     lambda a, t: has(a, "三年", "3 年", "3年")),
    ("文档·事实", "签订日期", ["协议是哪天签的"],
     lambda a, t: has(a, "2024") and has(a, "3 月 15", "03-15", "3月15")),
    ("文档·事实", "OCR 合同编号", ["扫描件里的合同编号是什么"],
     lambda a, t: has(a, "A-2024-001", "A-2024")),
    ("文档·事实", "产品响应时间", ["产品的响应时间目标是多少"],
     lambda a, t: has(a, "2s", "2 秒", "两秒")),

    # ---- 文档·计算（2）
    ("文档·计算", "总额×比例", ["若合同总额为 20 万元，违约金应赔付多少钱"],
     lambda a, t: has_amount(a, 10000)),
    ("文档·计算", "保证金加违约金", ["保证金和违约金加起来一共是多少钱"],
     lambda a, t: has_amount(a, 20000)),

    # ---- 文档·跨页（2）
    ("文档·跨页", "长文档上线时间", ["蓝鲸计划的上线时间是什么时候"],
     lambda a, t: has(a, "2026") and has(a, "9 月 30", "09-30", "9月30")),
    ("文档·跨页", "长文档项目代号", ["那个计划的项目代号叫什么"],
     lambda a, t: has(a, "蓝鲸")),

    # ---- 文档·结构化（3）
    # 注意：这份合成合同里没有甲方/乙方的具体名称，也没写合同总额数值。
    # 早先这一条要求抽出「甲方、乙方、金额」，模型如实回答「文档未提及」却被判失败——
    # 是任务问了一个不存在的字段，不是产品答错。改成文档里确实有的字段。
    ("文档·结构化", "字段抽取", ["把合作协议的关键字段抽出来：违约金比例、保证金、争议解决方式"],
     lambda a, t: has(a, "5%", "百分之五") and has(a, "10000", "一万") and has(a, "仲裁")),
    ("文档·结构化", "双文档对比", ["合作协议和产品手册在关键约定上有什么不同"],
     lambda a, t: has(a, "合作协议") and has(a, "产品手册")),
    ("文档·结构化", "摘要全部", ["把我上传的文档都总结一下"],
     lambda a, t: has(a, "违约", "合同") and has(a, "响应", "产品", "手册")),

    # ---- 文档·拒答（3）
    ("文档·拒答", "无关事实1", ["文档里写的服务器 IP 是多少"],
     lambda a, t: refused(a)),
    ("文档·拒答", "无关事实2", ["公司一共有多少名员工"],
     lambda a, t: refused(a)),
    ("文档·拒答", "无关事实3", ["甲方公司的统一社会信用代码是多少"],
     lambda a, t: refused(a)),

    # ---- 日常事务（8）
    ("日常", "当前时间", ["现在几点了"],
     lambda a, t: has(a, "点", "时", ":") and not asked_back(a)),
    ("日常", "加待办", ["帮我加个待办：周五前交实验报告"],
     lambda a, t: has(a, "待办", "添加", "已加", "记下")),
    ("日常", "看待办", ["我现在有哪些待办"],
     lambda a, t: has(a, "实验报告", "待办")),
    ("日常", "完成待办", ["实验报告那个待办我完成了"],
     lambda a, t: has(a, "完成", "已勾", "搞定", "标记")),
    ("日常", "记备忘", ["记一条备忘：路由器密码在抽屉第二层"],
     lambda a, t: has(a, "备忘", "记下", "已记")),
    ("日常", "查备忘", ["我的备忘里有什么"],
     lambda a, t: has(a, "抽屉", "密码", "备忘")),
    ("日常", "生成日报", ["生成今天的日报"],
     lambda a, t: has(a, "日报", "待办", "备忘", "天气")),
    ("日常", "设提醒", ["明天早上九点提醒我开会"],
     lambda a, t: has(a, "提醒", "设", "好") and not has(a, "需要补充")),
    ("日常", "制定计划", ["帮我制定一份考研复习计划"],
     lambda a, t: has(a, "计划", "阶段", "周", "复习")),

    # ---- 天气/联网（3）
    ("天气联网", "查天气", ["十堰明天天气怎么样"],
     # 第 5 次判分修正：模型写的是「℃」(U+2103)，判分只认 "°"(U+00B0)，白记一次失败。
     lambda a, t: has(a, "°", "℃", "度", "温")),
    ("天气联网", "联网搜索", ["联网查一下 DeepSeek 是什么公司"],
     lambda a, t: has(a, "DeepSeek", "深度求索", "幻方", "模型")),
    ("天气联网", "搜索存知识库", ["搜一下 LangGraph 是什么，并存到知识库"],
     lambda a, t: has(a, "LangGraph", "图", "状态机", "编排")),

    # ---- 澄清（5）：该问 vs 不该问
    ("澄清", "缺时间该问", ["帮我设个提醒"],
     lambda a, t: asked_back(a)),
    ("澄清", "缺内容该问", ["给我加个待办"],
     lambda a, t: asked_back(a)),
    ("澄清", "空泛该问", ["帮我搞一下"],
     lambda a, t: asked_back(a)),
    ("澄清", "不必问-时间", ["今天几号"],
     lambda a, t: not asked_back(a) and has(a, "月", "号", "日")),
    ("澄清", "不必问-待办", ["列出我的全部待办事项"],
     lambda a, t: not asked_back(a)),

    # ---- 多轮（4）
    ("多轮", "记住再计算", ["记住：我的合同总额是 20 万元", "按违约金 5% 算，我要赔多少"],
     lambda a, t: has_amount(a, 10000)),
    ("多轮", "指代追问", ["合作协议的违约金比例是多少", "那保证金呢"],
     lambda a, t: has_amount(a, 10000)),
    ("多轮", "条件更正", ["违约金按 5% 算，20 万的合同要赔多少", "不对，是 3%，重新算"],
     lambda a, t: has_amount(a, 6000)),
    ("多轮", "概要后追问", ["先告诉我上传了哪几个文档", "第一个文档的签订日期是什么时候"],
     lambda a, t: has(a, "2024")),

    # ---- 鲁棒（2）
    ("鲁棒", "不存在的文档", ["帮我总结一下《不存在的报告.pdf》"],
     lambda a, t: refused(a) or has(a, "没有", "未找到", "不存在", "没上传")),
    ("鲁棒", "空内容输入", ["？？？"],
     lambda a, t: len(a.strip()) > 0),
]


# ------------------------------------------------------------------ 执行

def prep() -> None:
    """准备语料与干净状态。"""
    import eval_docs

    main.bind_thread(THREAD)
    for name in DOCS:
        try:
            main.delete_document(name, thread_id=THREAD)
        except Exception:
            pass
    main.clear_todos()
    main.clear_memos()
    main.add_document("合作协议.pdf", eval_docs.build_contract_pdf(), thread_id=THREAD)
    main.add_document("产品手册.docx", eval_docs.build_manual_docx(), thread_id=THREAD)
    main.add_document("合同扫描件.png", eval_docs.build_scan_png(), thread_id=THREAD)
    main.add_document("长文档.pdf", eval_docs.build_long_pdf(), thread_id=THREAD)


SLEEP_BETWEEN = float(os.getenv("BENCH_SLEEP", "1.5"))  # 任务间节流，避开突发限流
TRANSIENT_RETRY = int(os.getenv("BENCH_RETRY", "2"))


def _transient(text: str) -> bool:
    """DeepSeek 突发调用会返回 400 Access denied，隔几秒自恢复。

    bench 必须分清「产品答错」和「接口抽风」，否则一次限流会把二十个任务记成失败。
    """
    return text.startswith("<异常") and ("Access denied" in text or "rate limit" in text.lower())


def run_task(turns: list) -> dict:
    """跑一个（可能多轮的）任务，返回成本与最终回答。"""
    graph = main.get_graph()
    before = main.usage_snapshot()
    t0 = time.perf_counter()

    ans = ""
    trace = ""
    retried = 0
    for text in turns:
        for attempt in range(TRANSIENT_RETRY + 1):
            try:
                res = graph.invoke(
                    {"messages": [HumanMessage(content=text)]},
                    config={"configurable": {"thread_id": THREAD}},
                )
            except Exception as e:  # noqa: BLE001
                ans = f"<异常：{type(e).__name__}: {e}>"
                if _transient(ans) and attempt < TRANSIENT_RETRY:
                    retried += 1
                    time.sleep(6 * (attempt + 1))
                    continue
                raise
            ans = res.get("execution_result") or ""
            trace = f"{res.get('plan', '')}\n{res.get('execution_trace', '')}\n{res.get('review', '')}"
            if _transient(ans) and attempt < TRANSIENT_RETRY:
                retried += 1
                time.sleep(6 * (attempt + 1))
                continue
            break
    dt = time.perf_counter() - t0
    after = main.usage_snapshot()

    return {
        "ans": ans,
        "trace": trace,
        "seconds": dt,
        "calls": after["calls"] - before["calls"],
        "tools": after["tool_calls"] - before["tool_calls"],
        "tin": after["in"] - before["in"],
        "tout": after["out"] - before["out"],
        "fails": after["fail"] - before["fail"],
        "retried": retried,
    }


def main_() -> int:
    print("=" * 72)
    print("任务级评测：一整句委托丢进状态图，看结果对不对 + 花了多少")
    print("=" * 72)

    only = os.getenv("BENCH_ONLY", "").strip()
    cases = [t for t in TASKS if only in t[1]] if only else list(TASKS)
    if only:
        print(f"（只跑名字含「{only}」的任务：{len(cases)} 条）")

    prep()
    print(f"语料就绪（{len(DOCS)} 份文档），开始跑 {len(cases)} 个任务…\n")

    rows = []
    for cat, name, turns, check in cases:
        try:
            r = run_task(turns)  # 内部已含突发限流重试
            ok = bool(check(r["ans"], r["trace"]))
        except Exception as exc:  # 任务级异常本身就是失败
            r = {"ans": f"<异常：{exc}>", "seconds": 0, "calls": 0, "tools": 0,
                 "tin": 0, "tout": 0, "fails": 1}
            ok = False
        r.update(cat=cat, name=name, ok=ok)
        rows.append(r)
        time.sleep(SLEEP_BETWEEN)  # 任务之间喘一口，别把突发限流打成失败
        mark = "✅" if ok else "❌"
        print(f"  {mark} [{cat}] {name:<12} {r['calls']} 调用 / {r['tools']} 工具 / "
              f"{r['tin']}+{r['tout']} token / {r['seconds']:.1f}s")

    # ---------------- 汇总
    n = len(rows)
    ok_n = sum(r["ok"] for r in rows)
    secs = [r["seconds"] for r in rows]
    calls = [r["calls"] for r in rows]
    toks = [r["tin"] for r in rows]

    def p(v, q):
        d = sorted(v)
        return d[min(len(d) - 1, int(round((q / 100) * (len(d) - 1))))]

    print("\n" + "=" * 72)
    print("一、总体")
    print("=" * 72)
    print(f"  任务成功率        {ok_n}/{n} = {ok_n / n * 100:.1f}%")
    print(f"  平均模型调用      {statistics.mean(calls):.1f} 次/任务")
    print(f"  平均工具调用      {statistics.mean([r['tools'] for r in rows]):.1f} 次/任务")
    print(f"  平均输入 token    {statistics.mean(toks):.0f}（中位 {statistics.median(toks):.0f}）")
    print(f"  平均输出 token    {statistics.mean([r['tout'] for r in rows]):.0f}")
    print(f"  延迟 P50/P95      {p(secs, 50):.1f}s / {p(secs, 95):.1f}s")

    print("\n" + "=" * 72)
    print("二、分类")
    print("=" * 72)
    print(f"  {'类别':<10}{'任务数':>6}{'成功':>6}{'成功率':>8}{'平均调用':>9}{'平均 token':>11}")
    cats = []
    for r in rows:
        if r["cat"] not in cats:
            cats.append(r["cat"])
    for c in cats:
        sub = [r for r in rows if r["cat"] == c]
        s_ok = sum(x["ok"] for x in sub)
        print(f"  {c:<10}{len(sub):>6}{s_ok:>6}{s_ok / len(sub) * 100:>7.0f}%"
              f"{statistics.mean([x['calls'] for x in sub]):>9.1f}"
              f"{statistics.mean([x['tin'] for x in sub]):>11.0f}")

    print("\n" + "=" * 72)
    print("三、成本结构")
    print("=" * 72)
    total_in = sum(r["tin"] for r in rows)
    total_out = sum(r["tout"] for r in rows)
    total_s = sum(secs)
    print(f"  40 个任务合计      {total_in} 输入 / {total_out} 输出 token，{total_s:.0f} 秒")
    print(f"  输入:输出         {total_in / max(1, total_out):.1f} : 1")
    print(f"  单位成本          每个任务 ≈ {total_in / n:.0f} token、{total_s / n:.1f} 秒")

    first, last = rows[:10], rows[-10:]
    print(f"  会话增长          前 10 个任务平均 {statistics.mean([x['tin'] for x in first]):.0f} token，"
          f"后 10 个平均 {statistics.mean([x['tin'] for x in last]):.0f} token"
          f"（{statistics.mean([x['tin'] for x in last]) / max(1, statistics.mean([x['tin'] for x in first])):.1f}×）")
    print(f"                    前 10 个平均 {statistics.mean([x['seconds'] for x in first]):.1f}s，"
          f"后 10 个平均 {statistics.mean([x['seconds'] for x in last]):.1f}s")

    failed = [r for r in rows if not r["ok"]]
    print("\n" + "=" * 72)
    print(f"四、失败清单（{len(failed)} 条）")
    print("=" * 72)
    if not failed:
        print("  无")
    for r in failed:
        ans = r["ans"].replace("\n", " / ")[:110]
        print(f"  ❌ [{r['cat']}] {r['name']}（{r['calls']} 调用 / {r['tools']} 工具）")
        print(f"     ↳ 实际回答：{ans}")

    # 清理
    for name in DOCS:
        try:
            main.delete_document(name, thread_id=THREAD)
        except Exception:
            pass

    return 0 if ok_n == n else 1


if __name__ == "__main__":
    sys.exit(main_())
