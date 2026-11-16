# 个人事务委托系统 —— 运行镜像
# 说明：运行期数据（agent_memory.db / sessions.json / plans.json / todos.json / kb / memos）
# 全部落在 DATA_DIR（默认 /app/data），通过 compose 挂载到宿主机，实现持久化。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data

WORKDIR /app

# rapidocr-onnxruntime 依赖 onnxruntime 与 opencv：
#   libgomp1   → onnxruntime 的 OpenMP 运行时
#   libgl1/libglib/libxcb/libx11 → opencv（非 headless 版）的动态库依赖，缺了 import cv2 会报 libxcb.so.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libgl1 libglib2.0-0 libxcb1 libx11-6 libsm6 libxext6 \
        fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# 依赖层单独一层，改代码不用重装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 只复制运行所需源码，.env / 数据库 / 虚拟环境都不会进镜像（见 .dockerignore）
COPY main.py a.py documents.py ./

RUN mkdir -p /app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status==200 else 1)"

CMD ["python", "-m", "streamlit", "run", "a.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
