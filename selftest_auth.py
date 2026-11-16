# -*- coding: utf-8 -*-
"""
访问口令网关自检（Streamlit AppTest，无需真正开服务）

用法：python selftest_auth.py
覆盖三种状态：
  1) 配置了口令 + 未登录  → 只能看到口令框，看不到聊天界面
  2) 配置了口令 + 口令错  → 报错，仍未登录
  3) 配置了口令 + 口令对  → 进入，且默认回到 user-main 主会话
  4) 没有配置口令（本机开发）→ 不锁死，但会话是随机 anon-*，不会串进 user-main
"""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(APP_DIR, "a.py")
PASSWORD = "selftest-pw-123"
_failed = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _failed.append(name)


def main():
    # ---------- 有口令 ----------
    os.environ["APP_PASSWORD"] = PASSWORD
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()

    print("[1] 未登录")
    check("显示口令输入框", len(at.text_input) > 0)
    check("不渲染聊天输入框", len(at.chat_input) == 0)
    check("未标记已登录", not at.session_state.get("authed"))

    print("[2] 口令错误")
    at.text_input[0].set_value("definitely-wrong")
    at.button[0].click().run()
    check("提示口令不正确", any("不正确" in e.value for e in at.error))
    check("仍未登录", not at.session_state.get("authed"))

    print("[3] 口令正确")
    at.text_input[0].set_value(PASSWORD)
    at.button[0].click().run()
    check("标记已登录", bool(at.session_state.get("authed")))
    check("进入 user-main 主会话", at.session_state.get("thread_id") == "user-main")
    check("渲染聊天输入框", len(at.chat_input) > 0)

    # ---------- 无口令（本机开发） ----------
    print("[4] 未配置口令")
    os.environ.pop("APP_PASSWORD", None)
    at2 = AppTest.from_file(APP, default_timeout=180)
    at2.run()
    tid = str(at2.session_state.get("thread_id") or "")
    check("不锁死，直接可用", len(at2.chat_input) > 0)
    check("默认会话是随机 anon-*", tid.startswith("anon-"), f"thread_id={tid!r}")

    print()
    if _failed:
        print(f"FAILED: {len(_failed)} 项 -> {_failed}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
