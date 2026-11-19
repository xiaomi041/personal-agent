# -*- coding: utf-8 -*-
"""提醒提示条自检（Streamlit AppTest，不需要真的起提醒服务）

用法：python selftest_alerts.py

为什么单独测这一块：以前「有未确认提醒」= 弹模态框 + st.stop()，一旦 ACK 发不出去
（提醒服务刚重启、8600 还没起来），那条提醒永远未确认，于是每次 rerun 都弹窗都 stop，
用户看到的就是「登录完只剩一个确认按钮，什么都点不了」。这个坑踩过三次，必须有回归。

覆盖：
  1) 有未确认提醒时，页面其余部分照常渲染（chat_input 在、不是只剩一个按钮）
  2) 提示条给出「确认 / 全部确认 / 10 分钟后再提醒」
  3) ACK 打不通时，本地仍把该条标为已读，界面不再锁死
  4) 「10 分钟后再提醒」只收起提示条，不丢提醒（侧边栏角标仍在）
"""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

import requests  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(APP_DIR, "a.py")
PASSWORD = "selftest-pw-123"
_failed = []

ALERTS = [
    {"id": 901, "content": "测试提醒：喝水", "fired_at": "2026-11-20 01:00:00"},
    {"id": 902, "content": "测试提醒：开会", "fired_at": "2026-11-20 01:05:00"},
]


class FakeResp:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def json(self):
        return self._payload


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _failed.append(name)


def install_fake_http(ack_fails: bool):
    """把 requests.get 换成假的：/alerts 返回固定两条，/ack 按参数成功或抛错。"""
    state = {"acked": []}

    def fake_get(url, *a, **kw):
        if "/alerts" in url:
            # 已确认的不再返回 —— 和真实服务语义一致
            left = [x for x in ALERTS if x["id"] not in state["acked"]]
            return FakeResp(left)
        if "/ack" in url:
            if ack_fails:
                raise requests.exceptions.ConnectionError("提醒服务没起来")
            aid = int(url.split("id=")[1])
            state["acked"].append(aid)
            return FakeResp({"ok": True})
        return FakeResp([])

    requests.get = fake_get
    return state


def login(at):
    at.text_input[0].set_value(PASSWORD)
    at.button[0].click().run()


def button_labels(at):
    return [b.label for b in at.button]


def main():
    os.environ["APP_PASSWORD"] = PASSWORD

    print("[1] 有未确认提醒，界面仍然可用")
    install_fake_http(ack_fails=True)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    login(at)
    check("登录成功", bool(at.session_state.get("authed")))
    check("聊天输入框还在（没被 st.stop 掐掉）", len(at.chat_input) > 0,
          f"chat_input={len(at.chat_input)}")
    check("页面没有异常", len(at.exception) == 0, [e.value for e in at.exception])
    labels = button_labels(at)
    check("有「确认」按钮", "确认" in labels, labels)
    check("有「全部确认」", "全部确认" in labels, labels)
    check("有「10 分钟后再提醒」", "10 分钟后再提醒" in labels, labels)
    check("提示条文案含条数", any("条提醒到点了" in str(m.value) for m in at.markdown),
          [str(m.value)[:40] for m in at.markdown])

    print("[2] ACK 打不通也能关掉（本地标为已读，不锁死）")
    idx = button_labels(at).index("确认")
    at.button[idx].click().run()
    check("点击后无异常", len(at.exception) == 0, [e.value for e in at.exception])
    check("仍是登录态", bool(at.session_state.get("authed")))
    check("聊天输入框仍在", len(at.chat_input) > 0)
    check("本地已标记该条", 901 in at.session_state.get("__alert_muted", set()),
          at.session_state.get("__alert_muted"))

    print("[3] 全部确认后提示条消失")
    if "全部确认" in button_labels(at):
        at.button[button_labels(at).index("全部确认")].click().run()
    check("提示条不再出现", not any("条提醒到点了" in str(m.value) for m in at.markdown))
    check("聊天输入框仍在", len(at.chat_input) > 0)
    check("无异常", len(at.exception) == 0, [e.value for e in at.exception])

    print("[4] 延后：收起提示条但不丢提醒")
    install_fake_http(ack_fails=True)
    at2 = AppTest.from_file(APP, default_timeout=180)
    at2.run()
    login(at2)
    labels = button_labels(at2)
    check("延后按钮存在", "10 分钟后再提醒" in labels, labels)
    at2.button[labels.index("10 分钟后再提醒")].click().run()
    check("延后后提示条收起",
          not any("条提醒到点了" in str(m.value) for m in at2.markdown))
    check("延后后仍提示还有几条（不静默丢弃）",
          any("未确认" in str(c.value) for c in at2.caption),
          [str(c.value)[:40] for c in at2.caption])
    check("延后后聊天输入框仍在", len(at2.chat_input) > 0)
    check("延后后无异常", len(at2.exception) == 0, [e.value for e in at2.exception])

    print()
    if _failed:
        print(f"失败 {len(_failed)} 项：" + "，".join(_failed))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
