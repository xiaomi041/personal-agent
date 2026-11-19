# -*- coding: utf-8 -*-
"""目标失败后换策略。

一条目标失败（模型说 ABORT、或续跑次数用尽）时，不直接判死——先判断"换一种做法有没有可能成功"。
这里只有**规则**，没有模型调用：换策略要快、要可解释、要能被测试钉死。
模型决策留给"计划本身怎么写"，"失败了要不要换、换哪一种"由这张表决定。

策略顺序（默认）：

    1. switch_tool  失败原因里点名了某个工具 → 换一个能顶上的工具重做
    2. ask_user     原因是缺关键信息（路径不存在、需要用户补参数）→ 转成向用户提问
    3. narrow       权限/路径/外部服务不可用 → 缩小目标，做能做到的部分并说明
    4. replan       其它情况 → 让规划节点换一条完全不同的路线
    5. giveup       以上都试过或次数用尽 → 放弃，并把试过什么写进结果

为什么 ask_user 不算"放弃"：它把状态从 active 改成 asked，用户回话就是新一轮输入——
对 agent 来说，问人也是完成任务的一种合法路径，而不是失败。
"""
import logging
import os
import re

log = logging.getLogger("agent.recovery")

# 最多换几次策略（不含第一次尝试）
MAX_RECOVERIES = int(os.getenv("GOAL_RECOVERY_MAX", "2"))

# 失败原因里出现这些词，说明"换个工具能救"
TOOL_SWAPS = {
    "ask_document": "别再让 ask_document 兜底，先用 search_documents 拿到原始片段，再自己组织答案并标注片段编号",
    "search_documents": "检索没命中，先用 summarize_document 看整体内容，再换关键词重试一次",
    "summarize_document": "全文摘要太贵太慢，改成先 search_documents 定位章节，只摘要命中部分",
    "write_file": "写文件失败时改用 outputs 目录下的纯文本回答，把内容直接写在回答里，让用户自己保存",
    "web_search": "联网失败就先用知识库里已有的条目回答，并明确说明这是本地资料、可能过期",
    "create_reminder": "建提醒失败就把时间和事项写进回答，让用户手动确认",
    "add_todo": "加待办失败就把待办清单直接打印出来，请用户复制粘贴",
}

# 提到这些词 = 缺信息，应该问人而不是硬做
_ASK_PATTERNS = (
    "文件不存在", "找不到文件", "未找到", "请提供", "缺少", "需要补充",
    "没有找到", "not found", "no such file", "无法读取",
)

# 提到这些 = 环境问题，缩小目标更合适
_NARROW_PATTERNS = (
    "权限", "permission", "denied", "超时", "timeout", "连接失败",
    "不可用", "无法访问", "服务异常", "余额", "配额",
)

STRATEGY_ORDER = ("switch_tool", "ask_user", "narrow", "replan")


class Recovery:
    """一次换策略的决定。"""

    def __init__(self, strategy: str, hint: str, reason: str = ""):
        self.strategy = strategy
        self.hint = hint
        self.reason = reason

    @property
    def giveup(self) -> bool:
        return self.strategy == "giveup"

    def as_dict(self) -> dict:
        return {"strategy": self.strategy, "hint": self.hint, "reason": self.reason}

    def __repr__(self) -> str:  # 便于日志与断言
        return f"Recovery({self.strategy!r}, {self.hint!r})"


def _mentioned_tool(text: str) -> str:
    for name in TOOL_SWAPS:
        if name in text:
            return name
    return ""


def pick(reason: str, tried: list | None = None, max_recoveries: int | None = None) -> Recovery:
    """根据失败原因与"已经试过哪些策略"给出下一步。

    tried 是策略名列表（按时间顺序）。返回的 strategy ∈ STRATEGY_ORDER ∪ {"giveup"}。
    """
    tried = list(tried or [])
    limit = MAX_RECOVERIES if max_recoveries is None else int(max_recoveries)
    text = (reason or "").strip()

    def unused(cands) -> str:
        for c in cands:
            if c not in tried:
                return c
        return ""

    # 次数用尽：无论还有什么没试，都不再折腾
    if len(tried) >= limit:
        return Recovery("giveup", "", f"已换过 {len(tried)} 次策略仍未成功")

    tool = _mentioned_tool(text)
    if tool and "switch_tool" not in tried:
        return Recovery("switch_tool", TOOL_SWAPS[tool], f"失败原因提到 {tool}")

    if any(p in text for p in _ASK_PATTERNS) and "ask_user" not in tried:
        return Recovery("ask_user", "把缺的具体信息整理成一句提问，直接问用户",
                        "failure 属于缺信息")

    if any(p in text.lower() for p in _NARROW_PATTERNS) and "narrow" not in tried:
        return Recovery("narrow", "缩小范围：先完成能做到的部分，明确说明哪一步没做成、为什么",
                        "failure 属于环境问题")

    nxt = unused(("replan", "switch_tool", "narrow"))
    if nxt:
        return Recovery(nxt, {
            "replan": "换一条完全不同的做法：换工具、换顺序、换分解方式，不要重复上次失败的步骤",
            "switch_tool": "试着绕开刚才那个工具，用其它工具组合达成同样目的",
            "narrow": "缩小目标，先交付能确定做到的部分",
        }[nxt], "没有可归因的具体线索，先换计划")

    return Recovery("giveup", "", "所有策略都试过了")


def hint_for_planner(rec: Recovery, reason: str) -> str:
    """拼成给规划节点看的提示。"""
    head = f"这条委托上一次没做成，失败原因：{reason or '未说明'}。"
    if rec.giveup:
        return head
    return f"{head}这次必须换一种做法：{rec.hint}。不要重复上次失败的步骤。"


def retry_state(rec: "Recovery", last_error: str = "") -> dict:
    """给 graph.invoke 补的字段：让规划节点和 executor 都知道"这是第几次换策略"。

    last_error 必须是**原始失败信息**（模型说的那句 / 异常的字符串），
    不能是 `rec.reason`（那是归类结论，写进 prompt 等于让模型猜上一次到底出了什么事）。
    """
    return {"retry_hint": "" if rec.giveup else rec.hint,
            "last_error": (last_error or rec.reason or "")[:300]}


def summarize(tried: list) -> str:
    """把尝试过的策略拼成一句人话，写进目标的 result。"""
    if not tried:
        return ""
    zh = {"switch_tool": "换工具", "ask_user": "改为提问", "narrow": "缩小范围", "replan": "换计划"}
    return "、".join(zh.get(t, t) for t in tried)


__all__ = [n for n in list(globals()) if not n.startswith("__")]
