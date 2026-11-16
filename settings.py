# -*- coding: utf-8 -*-
"""全局配置：密钥、路径、常量、日志。所有模块只依赖这里，这里不依赖任何业务模块。"""
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------- 密钥 ----------------
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
QWEN_KEY = os.getenv("QWEN_API_KEY", "")
if not DEEPSEEK_KEY or not QWEN_KEY:
    raise SystemExit("请在 .env 中配置 DEEPSEEK_API_KEY 和 QWEN_API_KEY")

# ---------------- 路径 ----------------
BASE_DIR = Path(__file__).parent
# 运行期数据目录：数据库/会话/计划/知识库/备忘都放这里。
# 本地默认就是项目根目录；容器部署时用 DATA_DIR 环境变量指向挂载卷，实现持久化。
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = DATA_DIR / "agent_memory.db"
MEMO_FILE = DATA_DIR / "memos.md"
TODO_FILE = DATA_DIR / "todos.json"
PLANS_FILE = DATA_DIR / "plans.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"

KB_DIR = DATA_DIR / "kb"
KB_CHUNKS = KB_DIR / "chunks.json"   # [{thread_id, doc, text}]
KB_VECS = KB_DIR / "vectors.npy"     # numpy 矩阵（行 = 片段）
KB_DOCS = KB_DIR / "docs.json"       # [{thread_id, name, chunk_count, created_at}]

# ---------------- 检索 / 模型配置 ----------------
EMBED_MODEL = "text-embedding-v3"
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

_GLOBAL_Q = re.compile(r"(总结|概括|归纳|全文|整体|大纲|结构|主要(讲|说|内容)|讲了(什么|啥)|全文要点)")

_SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# ---------------- 天气编码表 ----------------
_WMO_CODE = {
    0: "晴", 1: "大部晴朗", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇", 51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
    56: "冻毛毛雨", 57: "冻毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "冻雨", 71: "小雪", 73: "中雪", 75: "大雪",
    77: "雪粒", 80: "阵雨", 81: "强阵雨", 82: "暴雨",
    85: "阵雪", 86: "强阵雪", 95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴",
}
_WIND_DIRS = ["北风", "东北风", "东风", "东南风", "南风", "西南风", "西风", "西北风"]

# ---------------- 调度 ----------------
_WEEKDAY_MAP = {
    "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
    "mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1, "wed": 2,
    "wednesday": 2, "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4, "sat": 5, "saturday": 5, "sun": 6, "sunday": 6,
}
_scheduler_started = False
_ALERT_HTTP_PORT = int(os.getenv("ALERT_HTTP_PORT", "8600"))
_alert_server = None

# ---------------- 日志 ----------------
# 一行一个事件，全部走 stderr；容器里 docker logs 直接看得到。
# 级别由 LOG_LEVEL 控制（DEBUG 会打印工具入参与耗时明细）。
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    force=True,
)
# httpx 每发一次请求就 INFO 一行，会把真正的业务日志淹没，压到 WARNING。
# 注意 langchain 内部用的 logger 名是 "httpx2"，不是 "httpx"。
for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("agent")

__all__ = [n for n in list(globals()) if not n.startswith("__")]
