# Enterprise AI Agent · 企业级 AI Agent 服务

> 一个面向生产环境的企业级 AI Agent 框架：**FastAPI + LangGraph 状态图编排 + RAG + 多模型路由**。内置**意图识别 Pipeline（改写 → 分类 → 编排模式决策）**、**ReAct / Plan-Execute 统一状态图编排**、**计划台账与执行期收缩控制**、**人工审批（Human-in-the-loop）**、**短/长期记忆**、**混合检索 RAG**、**工具预算熔断**与**渐进式披露的高级技能（Skills）**，并配套 **评测 / 压测 / 监控** 三套工程化工具链。

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [状态图编排](#状态图编排)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [核心模块详解](#核心模块详解)
- [评测 / 压测 / 监控](#评测--压测--监控)
- [设计文档与规范](#设计文档与规范)
- [可观测性](#可观测性)
- [部署](#部署)
- [开发指南](#开发指南)
- [路线图 / 已知问题](#路线图--已知问题)
- [许可证](#许可证)
- [贡献](#贡献)

---

## 项目简介

本项目是一套**可运行、可扩展**的企业级 AI Agent 服务骨架，目标是把「大模型对话」升级为「能调用工具、能检索私有知识、能记住上下文、能按意图选择执行策略」的智能体。

它不是一个最小 demo，而是一套带有完整生产化考量的架构：

- **统一模型路由**：内置多模型路由器，支持单模型 / 多模型分层（FAST / STANDARD / DEEP）路由，带**熔断、健康探测与候选降级**。
- **意图前置决策**：每次请求先经过 3 阶段意图 Pipeline（查询改写 → 意图聚合分类 → 编排模式决策），再由状态图据此选择 `react` 或 `plan_execute`，并把「该用哪些工具、先调哪个工具、命中哪个知识库集合」一并下发，避免 Agent 在空想中盲目试错。
- **状态图编排**：ReAct 与 Plan-Execute 不再是两套并列的 if/else 分支，而是**同一张 LangGraph 状态图**上的两种拓扑——单节点自环（ReAct）与 计划→执行→重规划（Plan-Execute），节点级重试、断点续跑与人工审批因此成为图的原生能力。
- **执行期可控**：每一步都把**计划台账**（每个子任务的标识/目标/工具/状态/是否解决）渲染给模型，并允许它在证据充分时**跳过冗余子任务**或**提前收尾**——且不新增任何模型调用。
- **混合检索 RAG**：LlamaIndex 驱动的「向量（Milvus） + 关键词（BM25） + RRF 融合」混合检索。
- **联合记忆**：短期记忆（Redis 滑动窗口）+ 长期记忆（Milvus 向量召回），短期同步落库、长期异步沉淀，不阻塞主链路。
- **工具预算熔断**：单次请求内对工具调用做「单工具上限 / 累计无效 / 相关性抽查 / 全局总闸」四重约束，防止 Agent 死循环调用工具。

---

## 核心特性

| 能力 | 说明 |
|------|------|
| 多模型路由与韧性 | 单模型兼容 / 多模型分层路由；异步熔断、健康预选、按 Tier 超时与候选顺序自动降级 |
| 意图识别 Pipeline | 查询改写（多子问题拆分 + 目标锚点产出）、意图聚合分类、向量意图树召回、Plan/ReAct 模式决策 |
| 状态图编排 | LangGraph `StateGraph`：`prepare → plan → execute ⇄ replan → reflect → summarize → persist`，条件路由为无副作用纯函数 |
| 执行期控制协议 | 计划台账全貌可见、可跳过指定子任务、可在证据充分时提前收尾；控制指令**复用已有的结论提炼调用**，零额外模型开销 |
| 目标锚点 | 改写阶段产出 `agent_goal`（一句话、≤60 字），在 ReAct 系统段 / 规划提示词 / 子任务台账首行**三处注入同一份内容** |
| 人工审批（HITL） | 危险工具（写表 / 导报表）执行前 `interrupt` 挂起，图状态落 Redis checkpoint；审批后按 `run_id` 从断点恢复并 SSE 续流 |
| 检查点与续跑 | `langgraph-checkpoint-redis` 持久化图状态，支持运行快照查询与断点恢复；未配置 Redis 时自动降级为进程内内存后端 |
| 混合检索 RAG | Milvus 向量索引 + 内存 BM25 + RRF 融合重排；支持按「知识库集合」定向召回 |
| 联合记忆 | 短期（Redis 窗口/摘要）+ 长期（Milvus 向量）；并行装载、异步沉淀 |
| 工具系统 | 联网搜索（豆包直连，内置 Tavily 降级通道）、RAG 检索、知识图谱、Excel 读写与自然语言查询、报表导出、飞书多维表格、文件沙箱 |
| 预算熔断 | 单工具上限 / 累计无效 / 相关性抽查 / 全局总闸，约束 Agent 工具调用 |
| 高级技能 | `SKILL.md` 渐进式披露，命中技能后用 `allowed-tools` 精确号令工具集 |
| 评测 / 压测 / 监控 | 黄金集评测（意图 / RAG / 工具 / 答案质量）、locust 压测与熔断专项、Langfuse 成本日报与评分回推 |
| 可观测性 | 原生接入 Langfuse（Trace / Span / Generation），未配置则自动降级为 no-op |
| 工程化 | Dockerfile + docker-compose 一键拉起 app 与中间件，Pydantic v2 配置，loguru 日志，GitHub Actions 评测门禁 |

---

## 系统架构

```mermaid
flowchart LR
    Client([客户端]) -->|POST /api/v1/chat/with_agent · SSE| API[FastAPI 应用<br/>app/main.py]

    subgraph 接入与决策
      API --> Pipeline[AgentQueryIntentPipeline<br/>改写(含 agent_goal) → 分类 → 模式决策]
      Pipeline -->|意图 / slots / mode| Orchestrator[AgentOrchestrator<br/>薄门面]
    end

    subgraph 状态图编排
      Orchestrator --> Runner[GraphRunner<br/>checkpoint + 门面适配]
      Runner --> Graph{{LangGraph StateGraph<br/>prepare → plan → execute ⇄ replan<br/>→ reflect → summarize → persist}}
      Graph --> Deps[deps: per-request GraphDeps]
    end

    subgraph 能力层（经依赖注入消费）
      Deps --> Tools[ToolRegistry<br/>工具注册中心]
      Deps --> Memory[MemoryManager<br/>短期+长期]
      Deps --> Router[ModelRouter<br/>多模型路由/熔断]
      Deps --> Skills[SkillManager<br/>渐进式披露]
    end

    subgraph 工具与数据
      Tools --> RAG[RAGService<br/>向量+BM25+RRF]
      Tools --> Web[Web / Graph / Excel / Feishu / File]
      RAG --> MV[(Milvus)] & BM25[(内存 BM25)]
      Memory --> RD[(Redis)] & MVL[(Milvus 长期记忆)]
    end

    Router --> LLM[OpenAI 兼容 LLM<br/>单/多模型]
    Pipeline -.-> Router
```

> **关键设计**：运行时依赖（工具注册中心 / 模型路由 / 记忆 / 技能 / Trace / 配置）全部经 `RunnableConfig["configurable"]["deps"]` 注入，**不进入 state、不被 checkpoint 序列化**——这是"图状态可安全持久化"的前提。

**一次 `/chat/with_agent` 请求的端到端流程：**

1. 拉取短期对话历史，与意图 Pipeline **并行**发起长期向量记忆召回（高延迟，重叠执行）。
2. 同步运行意图 Pipeline（以 `asyncio.to_thread` 包装，保持 `query_intent` 同步代码零侵入）：
   - **查询改写**：多子问题拆分、术语映射，并产出本轮 `agent_goal`；
   - **意图聚合分类**：LLM 分类 + 向量意图树召回融合；
   - **模式决策**：规则化判断走 `react` 还是 `plan_execute`，并产出 `allowed_tools` / `first_tool_hint` / 知识库集合定向约束。
3. 系统意图（sys）命中时短路，直接走标准聊天核心，跳过昂贵 Agent 循环。
4. 否则交给 `AgentOrchestrator`：装配 `GraphDeps` 并委派 `GraphRunner` 驱动状态图，注入记忆、工具 Schema、技能提示词与工具预算。
5. 最终答案以 **SSE 流式**逐段吐出（`data: {"content": ...}` → `data: {"done": true, ...}`）；若途中命中危险工具，则推 `awaiting_approval` 事件并挂起，等待审批恢复。

---

## 状态图编排

### 拓扑

```
START → prepare ─┬─ should_plan ─→ plan ─→ execute ◀──────┐
                 └──────────────→ execute ──▲              │
                                        (ReAct 自环)       │
replan  ─→ execute | summarize                             │
reflect ─→ execute | summarize                             │
summarize ─┬─ 证据不足且仍有余量 ─→ replan ────────────────┘
           └─ 否则 ─→ persist → END
```

| 节点 | 职责 |
|------|------|
| `prepare` | 组装系统提示、技能摘要、记忆上下文，决定是否需要规划 |
| `plan` | 复用 `PlannerAgent.plan()` 产出子任务计划；注入目标锚点与子问题覆盖约束 |
| `execute` | 两条形态：**plan 形态**（逐子任务取参 → 审批闸门 → 执行工具 → 结论提炼）与 **ReAct 形态**（FC 原生协议为主，文本 `Thought/Action` 协议兜底）；执行前渲染计划台账 |
| `replan` | 复用 `PlannerAgent.replan()` 补取证据；产出新计划时重置跳过记录 |
| `reflect` | 可选的质量门（默认关闭）；不通过则回到 `execute` 重试，受节点重试上限约束 |
| `summarize` | 证据充分性判定 + 生成最终答案；判定不足且仍有余量则触发 `replan` |
| `persist` | 落库短期记忆、后台异步沉淀长期记忆 |

### 执行期控制协议

`execute` 节点在每次子任务执行前，把一份**计划台账**渲染进模型输入：

- 每行包含：子任务标识 / 该子任务要解决的问题 / 调用的工具（推理型标明无工具）/ 执行状态（已完成 / 正在执行 / 待执行 / 已跳过 / 失败） / 是否解决了它要解决的问题（是 / 部分 / 否）；
- 台账**由 `plan` + `subtask_results` 纯函数推导**，不维护可能漂移的第二份状态；
- `agent_goal` 作为台账块**表格上方的首行**，而非表格的一列（它是所有子任务共同的约束，不该逐行重复）。

子任务结论提炼那次调用改为**结构化输出**，在结论之外附带 `next_action`（继续 / 提前收尾）与 `skip_task_ids`——**复用一次本来就要发生的调用，不新增模型调用**。

> 为什么不用"注册一个 `end` 工具"：那需要模型额外发起一轮调用、会进入工具白名单（被误规划成业务子任务）、还要消耗工具预算与受审批影响。本项目此前刚因"控制类工具混入业务清单"下线过 `write_todos`。

**提前收尾不绕过质量闸门**：剩余子任务被归一为"全部跳过"，但证据充分性判定仍在 `summarize` 执行，判定不足时依旧会 `replan` 补取——这不是绕过质量，只是不再执行冗余步骤。

### 人工审批与检查点

危险工具（由 `AGENT_DANGER_TOOLS` 指定，默认 `sales_sql_write,sales_report_export_tool`）在**真正执行之前**命中闸门：

1. `interrupt(payload)` 抛出中断信号，LangGraph 把当前图状态与 pending interrupt 写入 checkpointer，`astream` 正常结束而非报错；
2. 本次工具**未执行、预算未扣**，`execute` 节点不返回任何 state 更新；
3. 客户端拿到 `awaiting_approval` 事件后，调用 `POST /agent/runs/{run_id}/approval` 提交审批决定，`GraphRunner.resume()` 从断点重建依赖并续跑，SSE 继续推送。

总开关默认关闭（`AGENT_APPROVAL_ENABLED=false`），此时该闸门直接放行，线上行为与未引入审批时完全一致。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | FastAPI、Uvicorn |
| Agent / LLM | LangGraph 1.0（StateGraph + `langgraph-checkpoint-redis`）、LangChain、OpenAI 兼容 API（支持 DeepSeek / Qwen / GLM / Kimi 等） |
| 模型路由 | 自研 `ModelRouter`：异步熔断 + 分层 Tier + 候选降级 |
| 向量库 | Milvus（`pymilvus`） |
| 缓存 / 检查点 | Redis（`redis.asyncio`）；图状态 checkpoint 亦落 Redis |
| 关系库 | PostgreSQL + SQLAlchemy（异步 `asyncpg`） |
| RAG | LlamaIndex、`rank_bm25`（BM25）、RRF 融合 |
| Embedding | DashScope `text-embedding-v3`（dim=1024，固定单模型，不进路由） |
| 配置与校验 | Pydantic v2、pydantic-settings |
| 文档处理 | `unstructured`、`pypdf`、LlamaIndex 解析器 |
| 知识图谱 | LightRAG（`infrastructure/knowledgebase/light_rag.py`） |
| 日志与韧性 | loguru、tenacity、httpx |
| 评测 / 压测 | pytest、pytest-asyncio、locust、自研黄金集与指标模块 |
| 可观测性 | Langfuse |
| 质量工具 | ruff、mypy |

---

## 目录结构

```
Enterprise_aiagent/
├── app/
│   ├── main.py                      # FastAPI 入口 + lifespan 并行初始化
│   ├── config.py                    # Pydantic-settings 配置（含 LLM 分层、状态图开关）
│   ├── api/
│   │   ├── routes/                  # chat / agent_runs / document / kownledgebase / health
│   │   └── depends/                 # FastAPI 依赖注入中心（含 Pipeline 与 GraphRunner 装配）
│   ├── core/
│   │   ├── agent/
│   │   │   ├── orchestrator.py      # 编排门面：装配 GraphDeps，委派 GraphRunner
│   │   │   ├── graph/               # LangGraph 状态图（本次重构核心）
│   │   │   │   ├── builder.py       #   拓扑编译 + 条件路由纯函数
│   │   │   │   ├── runner.py        #   运行门面（checkpoint / 挂起判定 / 恢复）
│   │   │   │   ├── state.py         #   AgentGraphState（TypedDict + reducer）
│   │   │   │   ├── checkpoint.py    #   Redis / 内存 checkpointer
│   │   │   │   ├── approval.py      #   人工审批闸门与载荷解析
│   │   │   │   ├── deps.py          #   每请求运行时依赖（不进 state）
│   │   │   │   └── nodes/           #   prepare / plan / execute / replan / reflect / summarize / persist
│   │   │   ├── planner.py           # 规划能力（被 plan / replan 节点复用）
│   │   │   ├── react_agent.py       # 工具结果后处理与工具接口协议（ReAct 执行逻辑已入图节点）
│   │   │   └── toolcall.py          # 工具调用统一结构（FC 与文本协议共用）
│   │   ├── rag/                     # RAGService / HybridRetriever / BM25 / parse
│   │   ├── memory/                  # 短期(Redis) / 长期(Milvus) / 管理器
│   │   ├── tools/                   # 工具注册中心 + 内置工具（builtin/）
│   │   ├── skill/                   # 技能管理器（渐进式披露）
│   │   ├── chat_recognizer/         # 意图识别（标准对话用）
│   │   └── backends/                # 虚拟/物理文件系统后端
│   ├── infrastructure/
│   │   ├── cache/   database/   embeddings/   knowledgebase/   trace/   vectordb/
│   ├── llm_model_router/            # 多模型路由器（熔断/选择/执行/校验）
│   ├── query_intent/                # 意图 Pipeline（改写/分类/决策/引导/结构化 Schema）
│   └── models/                      # Pydantic 领域模型与枚举
├── skills/                          # 高级技能（每个子目录一个 SKILL.md）
├── evals/                           # 评测：黄金集 + 指标 + 报告 + 门禁
│   ├── golden/                      #   意图 / RAG / 工具黄金集与语料清单
│   ├── runners/                     #   评测执行器（含记忆隔离）
│   ├── tools/                       #   语料入库、答案判分、纯度体检、token 估算
│   └── report.py                    #   统一评测入口
├── benchmark/                       # 压测：locust 场景 + mock LLM 服务 + 熔断专项
├── monitoring/                      # 监控：成本计算 + Langfuse 日报 + 评测评分回推
├── docs/                            # 使用文档（评测/压测/监控/优化纪要）与技术提案
├── openspec/                        # 规范驱动变更：proposals / designs / tasks / specs
├── 测试/                             # 状态图、控制协议、台账、目标注入等单测
├── rag_data/ rag_data_enterprise/ rag_data_graph/   # 示例语料（含图谱语料）
├── .github/workflows/eval.yml       # 评测门禁 CI
├── Dockerfile
├── docker-compose.yml               # app + 中间件（PostgreSQL/Redis/etcd/minio/Milvus）
├── requirements.txt
├── pyproject.toml
├── .env.example                     # 环境变量模板
└── README.md
```

---

## 快速开始

### 前置条件

- Python **3.11+**（Docker 镜像基于 3.12-slim）
- 一个 OpenAI 兼容的 LLM 网关 Key（如 DeepSeek / 通义千问 / GLM / Kimi）
- （可选）本地或容器化的 PostgreSQL / Redis / Milvus；纯对话可仅依赖 LLM + Redis

### 方式一：本地开发

```bash
# 1. 克隆并进入项目
git clone https://github.com/aaamodel/Ai_Agent_Frame.git Enterprise-aiagent
cd Enterprise-aiagent

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                    # 以可编辑模式安装包（pyproject 定义）

# 3. 配置环境变量
cp .env.example .env
# 至少填写 OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_LLM_MODEL

# 4. 启动 API（需先有 Redis；如需全栈 RAG 还需 Milvus / PostgreSQL）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. 健康检查
curl http://127.0.0.1:8000/api/v1/health
```

### 方式二：Docker Compose（推荐）

`docker-compose.yml` 同时编排了 **app + 全部中间件**（PostgreSQL / Redis / etcd / minio / Milvus）：

```bash
# 准备 .env（已包含连接中间件所需的服务名，如 REDIS_URL=redis://redis:6379/0）
cp .env.example .env

# 一次拉起应用与依赖
docker compose up -d --build

# 查看日志
docker compose logs -f app
```

> ⚠️ 首次启动 Milvus 及其依赖（etcd / minio）可能需要数十秒就绪。应用启动时已对 Redis / 技能树 / 意图向量索引做后台预热与容错；若 Milvus 尚未就绪，相关 RAG / 长期记忆功能会优雅降级，不影响基础对话。

只想要中间件、自己本地跑 app？把 `docker-compose.yml` 中的 `app` 服务注释掉，其余照常 `docker compose up -d` 即可（注意此时 `.env` 里的 `REDIS_URL` / `DATABASE_URL` / `MILVUS_HOST` 要改回 `localhost`）。

### 方式三：零成本验证（不需要中间件）

不接任何外部服务也能跑通工具链：

```bash
# 离线单测与黄金集体检
pytest 测试 -q
python -m evals.tools.check_golden_purity      # 黄金集纯度体检，有 FAIL 返回 1

# mock LLM 压测（无需模型 API）
python benchmark/mock_llm_server.py &
bash benchmark/run_bench.sh 20 2m --mode mock
```

细节见 [`docs/README.md`](docs/README.md)（评测 / 压测 / 监控总索引）。

---

## 前端控制台

仓库内含一个 React 前端控制台（`web/`），提供三个区：**对话**（含危险工具审批与降级提示）、
**文档**（RAG 集合管理）、**知识库**（图谱集合管理）。

```bash
# 开发：前端 dev server 经 vite proxy 调后端
uvicorn app.main:app --reload        # 终端 1
cd web && npm install && npm run dev # 终端 2 → http://localhost:5173

# 日常使用：构建后由后端托管，单进程、同源、无 CORS
cd web && npm run build              # 产出 web/dist/
uvicorn app.main:app                 # 访问 http://127.0.0.1:8000/
```

前端测试：`cd web && npm test -- --run`

设计文档见 `docs/superpowers/specs/2026-09-19-agent-console-frontend-design.md`，
实现计划见 `docs/superpowers/plans/2026-09-19-agent-console-frontend.md`，
运行说明与常见坑见 `web/README.md`。

## 配置说明

所有配置通过 `app/config.py`（Pydantic-settings）读取，优先级：**进程环境变量 > `./.env` > 默认值**。

> `.env` 语法注意事项（python-dotenv 硬性要求）：注释必须独占一行，禁止行尾尾随 `#`；含特殊字符的字符串整体用双引号包裹；`LLM_MODELS` 这类 JSON 必须整体包双引号。

### 模型与中间件

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `OPENAI_API_KEY` | — | OpenAI 兼容协议 API Key（必填） |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | 兼容网关 BaseURL（如 DeepSeek / DashScope 兼容模式） |
| `OPENAI_LLM_MODEL` | `deepseek-chat` | 单模型模式默认模型；3 个 Tier 默认复用 |
| `LLM_MODELS` | 空 | 多模型 JSON 数组 `[{"model_id","api_key?","base_url?","priority?","supports_thinking?"}]` |
| `LLM_TIER_FAST` / `_STANDARD` / `_DEEP` | 空 | 各 Tier 候选 model_id（逗号或 JSON 数组，按顺序降级） |
| `LLM_TIER_*_TIMEOUT_MS` | 40000 / 60000 / 90000 | 各 Tier 总调用超时（毫秒） |
| `DATABASE_URL` | `postgresql+asyncpg://...` | 异步 SQLAlchemy 连接串 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 连接串 |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Milvus 地址 |
| `MILVUS_KB_COLLECTION_NAME` | `knowledge_base_v3` | RAG 知识库集合名 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 空 | 配置后自动开启可观测性，留空则不启用 |

Embedding 模型固定为 DashScope `text-embedding-v3`（dim=1024），**不进 ModelRouter 路由**——这是设计约定，避免 Embedding 与 Chat 模型混用同一套熔断策略。

### 状态图与执行控制开关

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `AGENT_CHECKPOINT_BACKEND` | `redis` | 图状态 checkpointer 后端；`redis` 失败自动降级为内存 |
| `AGENT_CHECKPOINT_TTL_SECONDS` | `86400` | 检查点 TTL（秒） |
| `AGENT_CHECKPOINT_PREFIX` | `agent_cp` | Redis key 前缀（多服务共库隔离） |
| `AGENT_EVIDENCE_GATE_ENABLED` | `true` | 汇总阶段的证据充分性闸门；不足时触发重规划补取 |
| `AGENT_APPROVAL_ENABLED` | `false` | 危险工具人工审批总开关；关闭时闸门直接放行 |
| `AGENT_DANGER_TOOLS` | `sales_sql_write,sales_report_export_tool` | 需人工审批的工具名单（逗号分隔） |
| `AGENT_REFLECT_ENABLED` | `false` | 反思质量门；开启后不通过会回到 `execute` 重试 |
| `AGENT_REFLECT_MIN_SCORE` | `60` | 反思质量门通过分数线 |
| `AGENT_NODE_RETRY_MAX` | `1` | 节点级最大重试次数 |
| `ENABLE_SKILL_TOOL_GATING` | `true` | 命中技能时用其 `allowed-tools` 号令工具集 |
| `ENABLE_EMPTY_RESULT_REPLAN` | `true` | 数据源返回空结果时触发重规划 |

---

## API 接口

所有接口挂在 `API_PREFIX`（默认 `/api/v1`）下。

### 健康检查
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 轻量存活探针 |
| GET | `/api/v1/health/ready` | 就绪探针（检查数据库连通性） |

### 对话
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | 非流式标准对话（含意图识别、记忆召回与落库） |
| POST | `/api/v1/chat/stream` | SSE 流式标准对话 |
| POST | `/api/v1/chat/with_agent` | **Agent 智能对话入口**：意图 Pipeline + 状态图编排，最终答案 SSE 流式吐出 |

`/chat/with_agent` 请求体：
```json
{
  "query": "帮我分析上季度华东区销售额下滑的原因",
  "session_id": "sess_001",
  "strategy": "auto"          // auto | react | plan_execute（后两者强制覆盖 Pipeline 模式决策）
}
```

SSE 事件示例：
```
data: {"content": "根", "trace_id": "..."}
data: {"content": "据显示", "trace_id": "..."}
...
data: {"done": true, "status": "success", "session_id": "sess_001", "trace_id": "..."}
```

命中人工审批时，流以如下事件收尾（同一 `run_id` 继续审批）：

```
data: {"awaiting_approval": true, "run_id": "...", "approvals": [...], "done": true, "status": "awaiting_approval"}
```

### Agent 运行态（人工审批）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/agent/runs/{run_id}` | 查询运行快照：是否暂停 / 下一节点 / interrupt 载荷 / 使用了哪种模式 |
| POST | `/api/v1/agent/runs/{run_id}/approval` | 提交审批决定（`{"approved": true, "comment": ""}`）后从断点恢复，SSE 续流 |

> 恢复后的续跑中若再次命中危险工具，会再次推 `awaiting_approval` 事件，同一 `run_id` 继续审批。

### 文档 / 知识库
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/documents/upload` | 上传文档 → 解析/分块 → 写入 Milvus + BM25，并记录 PG 元数据 |
| GET | `/api/v1/documents` | 列出已入库文档元数据 |
| POST | `/api/v1/documents/kownledgebase/upload` | 上传 `.txt/.pdf` 并织入 LightRAG 知识图谱（注：路由名 `kownledgebase` 为历史拼写，已保留兼容） |
| POST | `/api/v1/documents/knowledgebase/upload-bulk` | 多文件批量上传（202 受理，后台异步解析织入） |

---

## 核心模块详解

### 1. LLM 模型路由器（`app/llm_model_router`）
- `ModelRouter` 接收分层配置 `AIModelProperties`（providers + chat tiers + selection）。
- 启动期 `ChatTierConfigValidator` 做 **Fail-Fast** 校验（3 个 Tier 齐全、候选非空、超时 >0、DEEP 至少 1 个 thinking 候选）。
- 异步三件套：`AsyncModelHealthStore`（三态熔断 + 健康预选）、`AsyncModelSelector`（按 Tier/Thinking/Purpose 选择）、`AsyncModelRoutingExecutor`（逐候选故障转移，超时按 Tier 集中查表）。
- 5 类 Purpose → Tier 映射：`planner→STANDARD`、`react→FAST`、`reflection→DEEP`、`intent_analysis→FAST`、`chat→STANDARD`。
- 对外提供 `await chat(...)` / `await chat_with_tools(...)` 两种形态；`purpose_hint` 等路由控制参数在进入 SDK 之前被剔除，不会漏传。

### 2. 意图识别 Pipeline（`app/query_intent`）
- **3 阶段**：`AgentCombinedRewriteIntentService`（改写+意图合并单次调用）→ `IntentResolver` / `AgentIntentAggregator`（分类聚合、向量意图树召回融合）→ `ModeDecider`（纯规则化模式决策，无 LLM）。
- 产出 `IntentContext`：`intent`、`slots`（含 `agent_goal`、`allowed_tools`、`first_tool_hint`、`top_kb_node.collection_names`、`per_sub_questions` 等），直接驱动状态图。
- 改写阶段的两条链路（组合调用主链路 / 组合失败后回退的两段式降级链路）解析统一收敛在父类，**归一化只写一处**。
- 全局单例 `ModelRouter` 经适配器复用为同步 `IntentLLMService`，零侵入复用同一套熔断。

### 3. 状态图编排（`app/core/agent`）
- `AgentOrchestrator` 是**薄门面**：并行预热技能树 + 记忆装载 → 装配每请求 `GraphDeps` → 委派 `GraphRunner`。
- `GraphRunner` 负责 checkpoint 生命周期、"是挂起还是结束"的判定、以及审批恢复；节点与路由互不耦合。
- 条件路由（`route_after_prepare` / `route_after_execute` / `route_after_replan` / `route_after_reflect` / `route_after_summarize`）全部是**无副作用纯函数**，可直接单测。
- 工具调用以**标准化拦截代理**拉平结构化 dict / 异常为 Observation 文本；原生 Function Calling 工具定义同步透传。
- 空业务结果触发重规划、"计划子任务全坏"触发重规划（均可开关）；成功后短期记忆同步落库、长期记忆后台异步沉淀（强引用防 GC，不阻塞响应）。
- 图状态可持久化 → 支持**断点续跑**与**人工审批**，长任务不再全靠内存。

### 4. RAG 服务（`app/core/rag`）
- `RAGService` 门面：`MilvusIndexManager`（LlamaIndex 向量存储）+ `BM25IndexBuilder`（内存关键词）+ `HybridRetriever`（向量 + BM25 + RRF）。
- `ingest_texts(...)` 写向量库并刷新 BM25；`seed_bm25()` 启动时从 Milvus 全量重建 BM25（失败不阻塞）。
- `retrieve_contexts(query, top_k, collection_names)` 是唯一对外检索出口，支持**按知识库集合白名单定向召回**（意图路由硬约束）。

### 5. 记忆系统（`app/core/memory`）
- 短期：`ShortTermMemory`（Redis 滑动窗口 + 摘要压缩）。
- 长期：`LongTermMemory`（Milvus 向量召回）。
- `MemoryManager` 以 `asyncio.gather` 并行装载短期/长期，对外提供 `get_context` / `get_relevant` / `append_turn`。

### 6. 工具系统 & 预算熔断（`app/core/tools`）
内置工具（经 `bootstrap_tools` 注册）：

| 工具 | 说明 |
|------|------|
| `web_search` | 联网搜索（豆包 API 直连；空结果/异常时代码层自动降级内部 Tavily 通道，该通道不注册、模型不可见） |
| `rag_knowledge_search` | 混合检索知识库 |
| `knowledge_graph_search` | 知识图谱检索 |
| `sales_sql_query` | 业务库（SQLite）**自然语言取数**：一句中文问题，Vanna 生成并执行 SQL，返回结构化结果（只读） |
| `sales_sql_write` | 业务库受约束写入：语义参数映射为 `UPDATE`，条件须唯一命中一行（写操作，默认走人工审批） |
| `sales_report_export_tool` | 销售分析报表导出为 `.xlsx`（写操作，默认走人工审批） |
| `feishu_bitable_tool` | 飞书多维表格 |
| `file_read_tool` / `file_list_tool` / `file_grep_tool` | 安全文件沙箱（虚拟模式）；`file_list_tool` 支持递归 `depth`（默认 3 层）与文件名 `pattern` 过滤 |

> 数据库工具（`database` / `describe_table` / `list_tables`）目前为可选，默认未注册（见 `init_tools.py` 注释块），可按需启用。

### ⚠️ 业务库取数引擎（Vanna）：口径变更后**必须**手动重训

销售业务数据存在 SQLite（`data/sales.db`），自然语言取数由 `sales_sql_query` 承载，
底层是 **Vanna**（0.x，已锁 `vanna>=0.7.3,<0.8`）。它有**两类**训练数据，介入方式完全不同：

| 训练数据 | 是否自动 | 说明 |
|---|---|---|
| **DDL**（表名 / 列名 / 类型 / `CHECK` 枚举） | ✅ **全自动** | 工具首次使用时 `SalesVanna.sync_ddl()` 从 `sqlite_master` 增量同步；改表结构后无需人工介入 |
| **口径**（如"ICP 达标 ＝ 员工规模 ≥ 200"、"赢单率 = 赢单数/(赢单数+输单数)"） | ❌ **需手动** | Vanna **无法**从 DDL 推出这类业务判据 |

**口径改动后（改了字段字典的"说明"/判据、或新增业务口径），必须执行：**

```bash
python -m app.core.sales_db.train_vanna   # 幂等：先移除旧 documentation 再重训
```

> 不重训的后果：模型写 SQL 时**不知道这个口径**，会自行假设阈值——例如把 ICP 判据
> 猜成 `员工规模 >= 100`，查询结果看起来正常、实际全错且难以察觉。
>
> 口径来源：`app/core/sales_db/knowledge.py` 会从 5 张"字段字典" sheet 中**只提取口径与判据**
> （列名/类型/枚举取值已被 DDL 覆盖，刻意丢弃，避免重复注入）。

**业务库的构建与导出：**

```bash
python -m app.core.sales_db.seed --rebuild   # 建表（带约束）+ 从 xlsx 导入 + 扩量
python -m app.core.sales_db.seed --export    # 导出 xlsx（xlsx 已降级为导出视图）
```

**⚠️ 规则文档走的是另一个入口，不要和上面的 Vanna 训练混起来。**

上面那个入口只写 **Vanna 的训练数据**（口径 → Vanna 的 ChromaDB）。
而"阶段流转规则 / 折扣权限 / 组合策略"是 **RAG 文档**（→ 项目的 Milvus 知识库
`sales_kb`，由 `rag_knowledge_search` 消费），两者的目的地与消费者都不同：

```bash
python -m app.core.sales_db.export_rule_docs   # 写出 raw_data/sales_kb_docs/*.md
```

导出后需**人工上传**到知识库集合 `sales_kb`（`POST /documents/upload`）。
业务数据不在知识库里，无需上传。

`ToolCallBudget` 提供四重熔断：单工具上限、累计无效次数、相关性抽查（第 N 次做 LLM 相关性判定）、全局总闸；熔断状态实时注入 Agent 提示词，引导其转向或作答。

### 7. 高级技能（`app/core/skill` + `skills/`）
- 每个技能是一个子目录，含 `SKILL.md`（YAML frontmatter：`name` / `description` / `allowed-tools` / `license` / `compatibility`）。
- `SkillManager` 扫描并解析元数据，向 System Prompt 注入**精简摘要**（渐进式披露）；Agent 命中相关技能后，用 `file_read_tool` 读取完整 `SKILL.md` 再执行。
- 编排层按 query 或 Pipeline 声明解析出技能，用其 `allowed-tools` 与已注册工具取交集**号令**本请求的工具候选，避免无关工具污染。
- 内置示例技能：`skills/sales-intelligence-assistant`（销售情报与分析助手）。

---

## 评测 / 压测 / 监控

三套工具链构成一条闭环：**评测定基线 → 压测定容量 → 监控守线上**。

| 板块 | 目录 | 解决什么问题 |
|------|------|-------------|
| **评测** | `evals/` | 效果好不好、改了有没有变差——意图准确率 / Recall@5 / Hit@5 / MRR / 工具调用成功率 / 关键参数命中率 / 单轮 token 与 P95 延迟 / 答案合格率 |
| **压测** | `benchmark/` | 抗多少并发、多快、多贵、熔断后多久切走——locust 场景 + mock LLM 服务 + 熔断切换专项 |
| **监控** | `monitoring/` | 上线后每天花多少钱、质量有没有掉——成本计算、Langfuse 日报、评测评分回推 |

```bash
# 全量评测（需 Milvus / Redis / 模型 API）
python -m evals.report --run-all

# 黄金集纯度体检（防止示例派生条目污染）
python -m evals.tools.check_golden_purity

# 压测与熔断专项
bash benchmark/run_bench.sh 50 10m --mode real
python benchmark/test_circuit_breaker.py --repeat 2 --fault hang

# Langfuse 成本日报
python monitoring/langfuse_daily_report.py --date 2026-09-13
```

评测链路使用一次性 `eval::` 前缀 session_id 并跑完清理，**保证记忆隔离**，避免 token 指标随运行次数膨胀。

四条贯穿三套工具的硬规则：① 拿不到就写「不可用」，不用估算值填；② 阈值是配置不是代码；③ 删表必须显式确认；④ 压测数字必须标注「本机单机环境，非生产环境数据」。

完整说明见 [`docs/README.md`](docs/README.md)。

---

## 设计文档与规范

本项目采用**规范驱动开发（spec-driven）**：变更先落规范，再落代码。

- **技术提案**：[`docs/05_技术提案_计划执行控制与目标锚点.md`](docs/05_技术提案_计划执行控制与目标锚点.md) —— 解释「计划台账为什么从状态推导而不落库」「为什么用控制协议而不是注册 `end` 工具」「`agent_goal` 为什么要三处注入同一份内容」等关键设计取舍。
- **变更规范**：[`openspec/`](openspec/) —— 每个变更包含 `proposal.md`（动机与影响）、`design.md`（设计决策与已否决备选）、`tasks.md`（可验证任务清单）、`specs/**/spec.md`（MUST 级验收条款与 Scenario）。
- **使用文档**：[`docs/`](docs/) —— 01 评测 / 02 压测 / 03 监控 / 04 优化纪要。
- **人工审批流程**：[`app/人工审核的逻辑与流程.md`](app/人工审核的逻辑与流程.md)。

---

## 可观测性

原生接入 **Langfuse**：在 `.env` 配置 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`（可选 `LANGFUSE_HOST`）后即自动开启，状态图节点、工具调用、LLM 生成均产生 Trace / Span / Generation。未配置时 `@observe` 自动退化为 no-op，零侵入、零报错。

`monitoring/push_eval_scores.py` 可把评测分数推送到 Langfuse，让「每次上线后的质量」和「日常成本 / 延迟」出现在同一张图上。

---

## 部署

### Docker（单镜像）
```bash
docker build -t enterprise-aiagent:latest .
docker run -d --name ea-app -p 8000:8000 \
  --env-file .env \
  enterprise-aiagent:latest
```

### docker-compose（全套）
见 [快速开始 · 方式二](#方式二docker-compose推荐)。`docker-compose.yml` 已包含 `app` 服务（基于 `Dockerfile` 构建），并 `depends_on` 中间件。请在 `.env` 中将 `REDIS_URL` / `DATABASE_URL` / `MILVUS_HOST` 指向 compose 服务名（`redis` / `postgres` / `milvus`）。

生产建议：为 `app` 增加 `restart: unless-stopped`、资源限制、与 Milvus 的健康探针；将 `.env` 中的密钥改为来自密钥管理服务。若启用人工审批，`AGENT_CHECKPOINT_BACKEND` 必须为 `redis`（多实例部署时内存后端无法跨实例恢复 `run_id`）。

---

## 开发指南

```bash
# 安装开发依赖（lint / type / test）
pip install -e ".[dev]"

# 代码风格与静态检查
ruff check app
ruff format app
mypy app

# 运行测试（状态图 + 评测 + 压测工具链）
pytest 测试 evals benchmark -q
```

项目约定：
- 配置一律走 `app/config.py`，不散落硬编码；复杂字段（LLM 列表 / Tier）以 `str` 存、以 `*_parsed` 属性取。
- 全局单例（ModelRouter / RAG / 技能树 / 意图向量索引 / GraphRunner）在 `lifespan` 中构建并挂到 `app.state`，请求期经 FastAPI `Depends` 注入。
- **运行时依赖不进图状态**：新增一个节点需要用到的依赖，请加到 `GraphDeps`，而不是 `AgentGraphState`——否则会污染 checkpoint 序列化。
- 图的条件路由写成**无副作用纯函数**，便于单测；节点是否暂停由 `runner` 依据 checkpoint 快照判定，与路由无关。
- Embedding 固定单模型，不进路由；Chat 模型统一走 `ModelRouter`。
- 长耗时 IO（embedding、Milvus、文件解析）尽量 `asyncio.to_thread` 化，避免阻塞事件循环。

---

## 路线图 / 已知问题

- 路由路径 `/documents/kownledgebase/upload` 存在历史拼写（`kownledgebase`），为兼容保留，新接入建议走 `/documents/knowledgebase/upload-bulk`。
- 数据库类工具默认未注册，需启用请取消 `init_tools.py` 中对应注释并补充会话工厂注入。
- `react_agent.py` 中 ReAct 的执行逻辑已迁入 `execute` 节点，该模块目前只保留工具结果后处理与工具接口协议，属待清理的历史资产。
- 部分设计文档与运行产物（个人材料、trace 转储、评测运行输出）已在 `.gitignore` 中排除，不随仓库发布。
- 长期记忆召回当前按 `session_id` 过滤，与短期记忆存在内容重叠，去重策略依赖指纹归一化（见 `app/core/memory/long_term.py`）。

---

## 许可证

本项目以 **MIT 许可证** 开源，详见 [LICENSE](LICENSE)。

---

## 贡献

欢迎 Issue / PR。提交前请运行 `ruff` 与 `pytest`，并在 PR 中说明改动动机与测试覆盖。
