# 🤖 眯哥委托智能体（Personal Agent）

基于 **LangGraph 多智能体状态图**的个人委托助手：用自然语言把日常事务（提醒、待办、计划、日报、文档分析、联网搜索等）直接委托给 AI，自动拆解任务、调度工具执行，并由独立评审节点校验结果后回答。

## ✨ 功能特性

- **🗣️ 自然语言委托** —— 模糊指令自动解析意图与参数，如「10 秒后提醒我喝水」自动创建定时提醒；多轮对话保持上下文连贯
- **🧠 Multi-Agent 三节点状态图** —— LangGraph 构建 `Planner → Executor → Critic`：
  - **Planner**（DeepSeek）：拆解用户委托，规划执行步骤
  - **Executor**（DeepSeek）：函数调用循环调度工具（单次最多 6 轮），记录执行证据链
  - **Critic**（Qwen）：独立评审执行结果，不合格自动返工重做（最多 2 轮），保证回答质量
- **💾 SQLite Checkpoint 断点续接** —— 对话状态持久化到本地 SQLite，服务重启不丢上下文；多会话独立记忆，互不串扰
- **📚 RAG 文档分析** —— 上传 PDF / Word / 图片 → 文本解析提取 → 段落切块（超长段按 350 字 + 40 字重叠续切）→ Embedding 向量化入库 → 语义相似度检索 → 注入上下文问答总结，**答案自动溯源到具体片段**（「依据：《文档》片段#N」）
- **⏰ 定时调度与提醒** —— 自建 `threading` 后台调度器（30 秒轮询 SQLite 提醒任务）+ 本地 HTTP 提醒服务（端口 8600），到点前端弹窗强制确认；支持一次性 / 周期提醒
- **📊 日报自动聚合** —— 一键生成今日日报：天气（按城市）+ 待办清单 + 备忘 + 知识沉淀，可定时每天自动生成
- **🌐 联网搜索 → 知识库** —— 内置搜索工具，查到的资料自动整理沉淀进知识库，后续可直接问答
- **🖥️ Streamlit 界面** —— 侧边栏会话管理（新建 / 回访 / 置顶 / 搜索 / 删除）+ 四分区工作台：会话 / 待办 / 任务计划 / 文档

## 🏗️ 系统架构

```
用户 ──> Streamlit（8501）
            │
            ▼
     ┌── LangGraph 状态图 ──────────────────────┐
     │  Planner（DeepSeek）任务拆解               │
     │      ↓                                    │
     │  Executor（DeepSeek）工具循环 ≤6 轮       │
     │      ↓                                    │
     │  Critic（Qwen）独立评审 ←── 不通过 → 返工 │
     │      ↓ 通过                                │
     └── 输出最终回答 ──────────────────────────┘
            │
   18 个工具 ┼── 备忘 / 待办 / 计划 / 天气 / 提醒 / 日报
            │── 文档问答（RAG） / 联网搜索（→知识库）
            └── 日期 / 写文件
```

**持久化层**：`agent_memory.db`（SQLite：对话 checkpoint + 提醒任务）、`sessions.json`（会话索引）、`plans.json`（任务计划）、`todos.json`（待办）、`kb/`（RAG 知识库：分块 + 向量）

## 🚀 快速开始

### 环境要求
- Python 3.10+

### 安装

```bash
git clone https://github.com/xiaomi041/personal-agent.git
cd personal-agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 配置密钥

```bash
cp .env.example .env
```

编辑 `.env` 填入两个 Key：

```
# DeepSeek API Key（规划 + 执行节点，https://platform.deepseek.com 获取）
DEEPSEEK_API_KEY=你的密钥

# 阿里云百炼 Qwen API Key（评审节点 + 文档向量化，https://bailian.console.aliyun.com 获取）
QWEN_API_KEY=你的密钥
```

### 启动

```bash
streamlit run a.py --server.port 8501
```

浏览器访问 `http://localhost:8501`

## 🎯 使用示例

| 说这句话 | 智能体会做什么 |
|---|---|
| 「10 秒后提醒我喝水」 | 解析时间参数，创建定时提醒，到点弹窗强制确认 |
| 「记一下：明天交房租」 | 写入备忘（`~/Desktop/memos.md`） |
| 「帮我规划备考」 | 自动拆解成多步任务计划，可逐步勾选完成 |
| 「生成今日日报」 / 「每天早上 8 点生成日报」 | 聚合天气 + 待办 + 备忘 + 知识沉淀，可定时 |
| 上传 PDF 后「总结这份文档」 | RAG 解析 → 向量检索 → 问答总结，带片段溯源 |
| 「明天十堰天气怎么样」 | 查询天气（含温度 / 风力 / 紫外线） |

## 📁 项目结构

```
personal-agent/
├── main.py            # 核心：状态图（Planner/Executor/Critic）+ 18 个工具 + RAG + 调度器
├── a.py               # Streamlit 界面（侧边栏会话管理 + 四分区工作台）
├── kb/                # RAG 知识库（chunks.json / vectors.npy / docs.json，按会话私有）
├── agent_memory.db    # SQLite：对话 checkpoint + 提醒任务（git 忽略）
├── sessions.json      # 会话索引（git 忽略）
├── plans.json         # 任务计划（git 忽略）
├── .env               # API 密钥（git 忽略，请勿上传）
├── .env.example       # 配置模板
├── requirements.txt   # 依赖清单
├── 启动.bat / 停止.bat # Windows 一键启停脚本（本地使用，git 忽略）
```

## 🛠️ 技术栈

- **LangGraph / LangChain**：多智能体状态图、工具调用循环、Checkpoint 持久化
- **Streamlit 1.65**：Web 界面
- **DeepSeek**：Planner / Executor 节点
- **Qwen（DashScope text-embedding-v3）**：Critic 评审节点 + 文档向量化
- **SQLite**：对话状态与提醒任务持久化
- **标准库 http.server / threading**：提醒 HTTP 服务与后台调度

## ⚠️ 免责声明

本项目为个人学习研究项目，请勿用于商业用途；生成内容可能不准确，重要事务请自行确认。

## 📄 License

MIT
