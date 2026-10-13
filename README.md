# 个人委托智能体（Personal Delegation Agent）

基于 **LangGraph** 的多智能体任务委托系统，Streamlit 前端。

## 功能

- **三节点状态图**：规划（DeepSeek）→ 执行（工具调用）→ 评审（Qwen 独立校验）
- **任务链**：任务拆解 + 进度跟踪
- **定时提醒 / 自动任务**：本地 HTTP 服务（端口 8600），到点弹窗提醒
- **会话管理**：多会话独立记忆、历史回访、删除、按会话私有的文档知识库
- **待办 / 日报 / 联网搜索 → 知识库 / RAG 文档问答**

## 快速开始

1. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 配置密钥：复制 `.env.example` 为 `.env`，填入 `QWEN_API_KEY`

3. 启动：

   ```bash
   streamlit run a.py --server.port 8501
   ```

4. 浏览器打开 <http://localhost:8501>

## 技术栈

- LangGraph / LangChain / SQLite（Checkpoint 持久化）
- Streamlit 1.65
- DeepSeek（规划/执行）· Qwen（评审）
