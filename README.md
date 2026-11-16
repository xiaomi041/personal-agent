# 🤖 眯哥委托智能体（Personal Agent）

基于 **LangGraph 多智能体状态图**的个人委托助手：用自然语言把日常事务（提醒、待办、计划、日报、文档分析、联网搜索等）直接委托给 AI，自动拆解任务、调度工具执行，并由独立评审节点校验结果后回答。

## ✨ 功能特性

- **🗣️ 自然语言委托** —— 模糊指令自动解析意图与参数，如「10 秒后提醒我喝水」自动创建定时提醒；多轮对话保持上下文连贯
- **🤔 澄清追问** —— 关键信息缺失时先反问（如「设个提醒」没说时间），不猜参数；能从工具拿到的信息（日期、天气、待办列表）不会反向来问用户
- **🧠 Multi-Agent 三节点状态图** —— LangGraph 构建 `Planner → Executor → Critic`：
  - **Planner**（DeepSeek）：拆解用户委托，规划执行步骤
  - **Executor**（DeepSeek）：函数调用循环调度工具（单次最多 6 轮），记录执行证据链
  - **Critic**（Qwen）：独立评审执行结果，不合格自动返工重做（最多 2 轮），保证回答质量
- **💾 SQLite Checkpoint 断点续接** —— 对话状态持久化到本地 SQLite，服务重启不丢上下文；多会话独立记忆，互不串扰
- **📚 文档分析（RAG）** —— 上传 PDF / Word / Markdown / 图片后走**结构化解析**：PDF 按页保留页码、按字号识别章节，Word 保留标题层级与表格，图片与扫描页自动 OCR（RapidOCR 本地识别，失败回退多模态模型）；切片携带 `页码 + 章节路径`，回答自动溯源到「第 N 页 · 某章节」；表格整块保留不切分，供数值类提问精准命中；「总结全文」类全局问题走 **map-reduce 分段摘要**，避免 top-k 检索漏内容；检索带相似度阈值与字面重合度加权，问不到就明确说「没有」而不是硬编；另提供**字段抽取**与**多文档对比**两个分析工具
- **📝 读 → 析 → 写 闭环** —— 文档分析/字段抽取的结果可直接导出 Word（.docx）或 PPT（.pptx），本地存桌面、容器内存 `DATA_DIR/outputs`
- **⏰ 定时调度与提醒** —— 自建 `threading` 后台调度器（30 秒轮询 SQLite 提醒任务）+ 本地 HTTP 提醒服务（端口 8600），到点前端弹窗强制确认；支持一次性 / 周期提醒
- **📊 日报自动聚合** —— 一键生成今日日报：天气（按城市）+ 待办清单 + 备忘 + 知识沉淀，可定时每天自动生成
- **🌐 联网搜索 → 知识库** —— 内置搜索工具，查到的资料自动整理沉淀进知识库，后续可直接问答
- **🖥️ Streamlit 界面** —— 侧边栏会话管理（新建 / 回访 / 置顶 / 搜索 / 删除）+ 四分区工作台：会话 / 待办 / 任务计划 / 文档

## 📄 文档分析是怎么做的

```
上传文件
   │
   ├─ PDF   → PyMuPDF 按页取文字（保留页码）+ 字号识别章节 + pdfplumber 抽表格
   ├─ Word  → 按正文顺序遍历段落与表格（不丢表格）+ Heading 样式维护章节路径
   ├─ 图片  → RapidOCR 本地识别，失败回退 Qwen-VL
   └─ 扫描页 → 栅格化(200dpi) 后走 OCR
   │
   ▼
切片：表格整块保留；正文按段落，>350 字在句末标点处断开，40 字重叠
    每个切片带 { doc, page, heading, kind }
   │
   ▼
检索：0.7 × 向量余弦 + 0.3 × 字符 BM25（精确词不被向量稀释）
     低于阈值(默认 0.42) 直接不返回 —— 宁可说「没有」，也不硬凑
   │
   ▼
回答：普通问题 top-k 注入；「总结全文」走 map-reduce（分批摘要 → 归并）
     引用格式：《文档》· 第 N 页 · 章节名
```

**为什么这么做**

| 设计 | 解决的问题 |
|---|---|
| 表格单独成块、不参与定长切分 | 定长切分会把表格切成半行，数值类提问直接失效 |
| 切片带页码 + 章节 | 「片段#3」无法回到原文，页码+章节才能定位 |
| BM25 混合检索 | 「合同编号 A-2024-001」这类精确词，纯向量会被相似但无关的段落挤掉 |
| 相似度阈值 | 没有阈值时，任何问题都能凑出 5 个片段，模型只能硬编 |
| map-reduce 摘要 | top-k 只能覆盖文档一小部分，「总结全文」必然漏内容 |
| 内容哈希判重 | 重复上传同一文件会重复消耗 embedding 费用 |

**评测**：`eval_docs.py` 内置 17 条问题（含 12 页长文档、扫描件 OCR、表格数值、多步计算、4 条不可答），
覆盖检索 top-k 命中率、答案正确率、拒答准确率、引用准确率四项指标。

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
   20 个工具 ┼── 备忘 / 待办 / 计划 / 天气 / 提醒 / 日报
           │── 文档分析（问答 / 字段抽取 / 对比） / 联网搜索（→知识库）
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
| 上传 PDF 后「总结这份文档」 | 结构化解析 → 分段摘要（map-reduce）→ 带页码溯源的总述 |
| 「提取甲方、金额、保密期限」 | 字段抽取成表格（字段 / 内容 / 出处），缺失字段标「未提及」 |
| 「对比这两份合同的责任条款」 | 多文档差异表格 + 一句话总结 |
| 「把上面的结论导出成 Word」 | 生成 `.docx` 到桌面（容器内到 `DATA_DIR/outputs`） |
| 「明天十堰天气怎么样」 | 查询天气（含温度 / 风力 / 紫外线） |

## 📁 项目结构

```
personal-agent/
├── main.py            # 核心：状态图（Planner/Executor/Critic）+ 20 个工具 + 调度器
├── documents.py       # 文档解析引擎：PDF/Word/Markdown/图片 → 结构化块与切片
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
- **PyMuPDF / pdfplumber / RapidOCR**：PDF 分页与表格抽取、图片 OCR
- **SQLite**：对话状态与提醒任务持久化
- **标准库 http.server / threading**：提醒 HTTP 服务与后台调度

## 🐳 部署（Docker）

```bash
docker build -t personal-ai .
docker run -d --name personal-ai \
  -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  --restart unless-stopped \
  personal-ai
```

- 运行期数据（`agent_memory.db` / `sessions.json` / `plans.json` / `todos.json` / `kb/` / `memos.md`）统一落在 `DATA_DIR`（容器内 `/app/data`），挂载卷后重建容器不丢。
- 密钥通过 `--env-file` 注入，`.dockerignore` 已排除 `.env`，不会打进镜像。
- 与火车票项目同时部署时使用根目录的 `docker-compose.yml`；公网访问（natapp 内网穿透）见 `DEPLOY.md`。

## ⚠️ 免责声明

本项目为个人学习研究项目，请勿用于商业用途；生成内容可能不准确，重要事务请自行确认。

## 📄 License

MIT
