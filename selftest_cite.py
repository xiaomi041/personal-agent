"""引用校验的离线自检：不联网，用假模型把三条分支都走到。

为什么要单独测：线上跑 bench_cite 时模型很配合（12 次全 pass，一次纠正都没触发），
也就是说「纠正」和「降级」这两条分支在真实流量里几乎不会自己出现——
不主动造用例，就等于没写。

运行：python selftest_cite.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DATA_DIR", os.path.join(HERE, ".tmp-cite-selftest"))
sys.path.insert(0, HERE)

import knowledge  # noqa: E402
import runtime  # noqa: E402

PASS = 0
FAIL = 0

HITS = [
    {"doc": "合作协议.pdf", "page": 1, "heading": "第一章 总则", "kind": "text",
     "text": "违约金为合同总额的 5%。", "score": 0.8, "cosine": 0.8, "bm25": 0.5, "idx": 0},
    {"doc": "合作协议.pdf", "page": 2, "heading": "第二章 争议", "kind": "text",
     "text": "协商不成的，提交武汉仲裁委员会仲裁。", "score": 0.5, "cosine": 0.5, "bm25": 0.3, "idx": 1},
]


def ok(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


class _Resp:
    def __init__(self, content: str):
        self.content = content


def run_ask(fake_replies, query="违约金比例是多少"):
    """用假模型跑一次 ask_document，返回 (输出, 调用次数)。"""
    calls = {"n": 0}

    def fake_call_llm(model, messages, **kw):
        i = min(calls["n"], len(fake_replies) - 1)
        calls["n"] += 1
        return _Resp(fake_replies[i])

    orig = (knowledge.call_llm, knowledge.search_documents, knowledge._thread_chunks)
    knowledge.call_llm = fake_call_llm
    knowledge.search_documents = lambda q, **kw: list(HITS)
    knowledge._thread_chunks = lambda t: list(HITS)
    try:
        fn = getattr(knowledge.ask_document, "func", knowledge.ask_document)
        return fn(query), calls["n"]
    finally:
        knowledge.call_llm, knowledge.search_documents, knowledge._thread_chunks = orig


def main() -> int:
    print("=" * 72)
    print("引用校验自检")
    print("=" * 72)

    # ---- 纯函数 ----
    ok("解析编号 [1][2]", knowledge._cited_ids("结论 [1]，另一条 [2]。", 5) == [1, 2])
    ok("忽略越界编号", knowledge._cited_ids("结论 [9]", 2) == [])
    ok("编号去重", knowledge._cited_ids("A [1] B [1]", 5) == [1])
    ok("无编号返回空", knowledge._cited_ids("没有引用的回答", 5) == [])
    ok("识别拒答", knowledge._is_refusal("文档未提及"))
    ok("识别检索失败也算拒答", knowledge._is_refusal("知识库中没有检索到与这个问题相关的内容"))
    ok("正常回答不算拒答", not knowledge._is_refusal("违约金是 5% [1]"))

    ok("有引用 → 可追溯", "可追溯" in knowledge._citation_note("违约金 5% [1]", HITS))
    ok("无引用 → 警告", "⚠" in knowledge._citation_note("违约金 5%", HITS))
    ok("拒答 → 专项提示", "没有答案" in knowledge._citation_note("文档未提及", HITS))

    # ---- 分支 1：模型自己标了编号 ----
    before = runtime.usage_snapshot()["cite_pass"]
    out, n = run_ask(["违约金为合同总额的 5% [1]。"])
    ok("分支1 直接用原回答", "5%" in out and "可追溯" in out, out[:60])
    ok("分支1 只调一次模型", n == 1, f"calls={n}")
    ok("分支1 计入 pass", runtime.usage_snapshot()["cite_pass"] == before + 1)

    # ---- 分支 2：首答没编号，纠正后达标 ----
    before = runtime.usage_snapshot()["cite_repaired"]
    out, n = run_ask(["违约金是百分之五。", "违约金为合同总额的 5% [1]。"])
    ok("分支2 采用纠正后的回答", "5%" in out and "[1]" in out, out[:60])
    ok("分支2 调了两次模型", n == 2, f"calls={n}")
    ok("分支2 计入 repaired", runtime.usage_snapshot()["cite_repaired"] == before + 1)

    # ---- 分支 3：两次都没依据 → 降级为未提及 ----
    before = runtime.usage_snapshot()["cite_rejected"]
    out, n = run_ask(["违约金是百分之五。", "大概是五个点吧。"])
    ok("分支3 降级为未提及", "未提及" in out and "已降级" in out, out[:80])
    ok("分支3 计入 rejected", runtime.usage_snapshot()["cite_rejected"] == before + 1)

    # ---- 分支 4：模型直接拒答 → 不做纠正 ----
    out, n = run_ask(["文档未提及"])
    ok("分支4 拒答不触发纠正", n == 1 and "没有答案" in out, f"calls={n}")

    print("-" * 72)
    print(f"通过 {PASS} / 失败 {FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
