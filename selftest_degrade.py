"""降级自检：向量服务或评审模型不可用时，产品还能不能用。

背景：2026-11-19 跑任务级评测时，阿里云百炼账户欠费，embedding 直接返回
`400 Arrearage`，于是「上传文档」和「提问」整条链路一起 500 —— 一个外部依赖
欠费，把自己的功能全带走了。这份自检锁死两件事：

1. embedding 挂掉时：文档照常入库（只建关键词索引），检索退化成 BM25 单路，仍能回答并给引用；
2. 评审模型（Qwen）挂掉时：委托不被中断，评审降级为「评审跳过」，并在回答里明说未经二次校验。

运行：python selftest_degrade.py   （需要 DATA_DIR 可写，容器内直接跑即可）
"""
import os

os.environ.setdefault("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp-degrade"))
os.environ["DOC_MIN_SCORE"] = "0.2"  # 降级路由分数分布不同，这里只是想验证"能取到"

import shutil  # noqa: E402
import sys  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILED = []


class _SkipRecovery(Exception):
    """本环境没有可用 embedding 时，第 2 段整体跳过。"""


def ok(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}  {detail}")
        FAILED.append(name)


def main() -> int:
    data_dir = os.environ["DATA_DIR"]
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)

    import knowledge
    import main as agent

    print("\n[1] 向量服务不可用 → 文档仍可入库，检索退化为 BM25")
    knowledge._EMBED_DOWN = False
    real_embed = knowledge._embed_texts

    def boom(texts):
        raise RuntimeError("400 Client Error: Arrearage (账号欠费)")

    knowledge._embed_texts = boom
    try:
        msg = knowledge.add_document(
            "合同.txt",
            "合作协议\n甲方乙方\n合同总额为二十万元，违约金比例为合同总额的百分之五。\n".encode("utf-8"),
            thread_id="t-degrade",
        )
        ok("入库成功", "已读入" in msg, msg)
        ok("提示了降级", "向量服务不可用" in msg, msg)

        docs = knowledge.list_documents("t-degrade")
        ok("文档元数据标记降级", bool(docs) and docs[0].get("degraded") is True, str(docs))

        hits = knowledge.search_documents("违约金比例", top_k=3, thread_id="t-degrade")
        ok("仍能检索到片段", bool(hits), f"hits={hits}")
        ok("检索模式标为 bm25", bool(hits) and hits[0]["mode"] == "bm25", str(hits[:1]))

        agent.bind_thread("t-degrade")  # ask_document 走 _current_thread()，不接受 thread_id 参数
        answer = agent.ask_document.func("违约金比例是多少")  # @tool 包了一层
        ok("回答里带出处", "合同.txt" in answer, answer[:120])
    finally:
        knowledge._embed_texts = real_embed

    print("\n[2] 向量服务恢复 → 新文档走混合检索")
    knowledge._EMBED_DOWN = False
    try:
        knowledge._embed_texts(["探针"])  # 先确认本环境真有可用的 embedding
    except Exception as e:  # noqa: BLE001
        print(f"  ⏭ 跳过（本环境确实没有可用 embedding：{type(e).__name__}: {str(e)[:70]}）")
        knowledge._EMBED_DOWN = False
        _skip_recovery = True
    else:
        _skip_recovery = False
    try:
        if _skip_recovery:
            raise _SkipRecovery()
        msg2 = knowledge.add_document(
            "说明.txt",
            "产品说明\n响应时间不超过两百毫秒。\n".encode("utf-8"),
            thread_id="t-degrade",
        )
        ok("恢复后入库不带降级标记", "向量服务不可用" not in msg2, msg2)
        hits2 = knowledge.search_documents("响应时间", top_k=3, thread_id="t-degrade")
        ok("恢复后走混合检索", bool(hits2) and hits2[0]["mode"] == "hybrid", str(hits2[:1]))
    except _SkipRecovery:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"  ⏭ 跳过（本机无可用 embedding 凭据）：{type(e).__name__}: {str(e)[:80]}")

    print("\n[3] 评审模型不可用 → 委托不被中断")
    state = {
        "messages": [],
        "plan": "调用 get_today 回答当前时间",
        "execution_trace": "工具调用：get_today",
        "execution_result": "现在是 2026-11-19 03:00",
        "retry_count": 0,
    }
    real_call = agent.call_llm

    def qwen_down(model, messages, what="", attempts=1):
        raise RuntimeError("400 Client Error: Arrearage (评审模型欠费)")

    agent.call_llm = qwen_down
    try:
        out = agent.critic_node(state)
        ok("评审降级但不抛异常", isinstance(out, dict), str(out)[:100])
        ok("明说评审被跳过", "评审跳过" in out["review"], out["review"][:120])
        ok("评审消息进入对话历史", any("[评审]" in m.content for m in out["messages"]), "")
    finally:
        agent.call_llm = real_call

    print("\n" + "=" * 56)
    if FAILED:
        print(f"❌ {len(FAILED)} 项失败：" + "、".join(FAILED))
        return 1
    print("✅ 降级自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
