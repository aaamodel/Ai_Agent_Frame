# Agent 控制台前端 设计文档

**日期：** 2026-09-19
**状态：** 待审阅
**范围：** 新增前端子系统（对话 / 文档 / 知识库三个区），一次交付

---

## 1. 目标与用途

为 `Enterprise_aiagent` 后端 `app/api/routes/` 提供的 API 建一个 **Web 前端控制台**。

**用途定位：内部日常工具，会长期使用、持续加功能。**

这一条决定了后面所有结构决策的重量级：
- 需要多页面路由（不是单页切换）
- 需要清晰的数据层（服务端状态有缓存与失效策略）
- 目录结构要可扩展，加一个新功能区不应该牵动已有代码

**成功标准：**
1. 能在浏览器里完成一次完整对话（含流式回答、危险工具审批、降级提示）
2. 能查看与管理 RAG 集合及其文件（上传、列表、删除）
3. 能查看与管理知识图谱集合及其文件
4. 后端一条命令起来后，前端同源可用，无跨域配置
5. 加一个新功能区（例如"审批中心"）时，改动局限在新目录 + 左栏一行

---

## 2. 后端现状（已核实的事实，非假设）

这些是设计的地基。每条都实际读过代码或跑过命令确认。

### 2.1 路由与前缀

API 前缀 **`/api/v1`**（`app/config.py:178`）。五个 router 均由 `app/main.py:475-479` 挂载：
`health` / `chat` / `agent_runs` / `document` / `kownledgebase`。

### 2.2 对话接口与 SSE 协议

`POST /api/v1/chat/with_agent`，请求体 `{query, session_id, strategy}`，返回 **SSE**。

事件格式（`app/api/routes/chat.py:296-298` 定义的 `_sse_payload`，每条 `data: <json>\n\n`）：

| 事件 | 载荷 | 含义 |
|---|---|---|
| 正文片段 | `{"content": "…", "trace_id": "…"}` | 逐段吐出最终答案（打字机） |
| 正常结束 | `{"done": true, "status": "success"\|"degraded", "session_id": …, "trace_id": …, "steps_executed": N, "degraded": bool}` | 本轮结束 + 元数据 |
| **待审批** | `{"awaiting_approval": true, "run_id": …, "trace_id": …, "approvals": […], "done": true, "status": "awaiting_approval"}` | 命中危险工具，**图已挂起** |
| 失败 | `{"error": "…"}` | 本轮失败 |

⚠️ **`awaiting_approval` 是阻塞语义**：事件发出后本轮流即结束，**没有答案**。
必须由前端调审批接口恢复，否则这次对话永久卡住。续跑中若再次命中危险工具，
会**再次**推 `awaiting_approval`（同一个 `run_id`），因此审批是一个**循环**，不是一次性。

### 2.3 审批接口

| 接口 | 说明 |
|---|---|
| `GET /api/v1/agent/approvals/pending?limit=N` | 列出全部挂起 run。返回 `backend`（`redis` / `memory`）+ 每条：`run_id` / `session_id` / `trace_id` / 原始问题 / 意图 / 模式 / 计划进度 / 暂停时间 / 工具名 / 参数预览 / `subtask_id` / 模型思考 |
| `GET /api/v1/agent/runs/{run_id}` | 运行快照：`exists` / `paused` / `next` / `interrupts` / `mode_used` / `success`。**不含完整执行步骤** |
| `POST /api/v1/agent/runs/{run_id}/approval` | 请求体 `{approved: bool, comment: str}`，返回 **SSE 续流**，协议与上表一致；`run_not_found` 错误码表示检查点已失效 |

**挂起态存储：** `app/config.py:311` `agent_checkpoint_backend` 默认 **`redis`**，失败自动降级 `memory`。
受 `agent_checkpoint_ttl_seconds` 限制。实际生效值写在 `app.state.agent_checkpoint_backend`（`app/api/depends/dependencies.py:482`）。

→ **结论：挂起项能跨后端重启存活（Redis 可用时），因此"刷新恢复"与"跨会话找回"是真实可用功能。**

### 2.4 文档（RAG）接口

| 接口 | 说明 |
|---|---|
| `POST /api/v1/documents/upload` | Form: `file` / `collection_name` / `description`；同步返回 `DocumentUploadResponse`（`document_id` / `filename` / `status` / `chunk_count` / `collection_name` / `description` / `message`） |
| `GET /api/v1/documents` | 全部文档元数据 |
| `GET /api/v1/vector/collections` | RAG 集合列表 |
| `GET /api/v1/vector/collections/{c}/files` | 某集合的全部文件（含各自切片数） |
| `DELETE /api/v1/vector/collections/{c}/files/{filename:path}` | 删除某文件的全部向量与元数据（**支持中文，需 URL 编码**） |

### 2.5 知识库（图谱）接口

| 接口 | 说明 |
|---|---|
| `POST /api/v1/documents/kownledgebase/upload` | 单文件，同步。Form: `file` / `collection_name` / `description` |
| `POST /api/v1/documents/knowledgebase/upload-bulk` | 多文件，**202 后台任务**，返回 `{task_id, …}` |
| `GET /api/v1/knowledgebase/collections` | 图谱集合列表（`name` / `legacy` / `description` / `document_count` / `files`） |
| `GET /api/v1/knowledgebase/collections/{c}/files` | 某图谱集合的文件（含 LightRAG 处理状态与切片数） |
| `DELETE /api/v1/knowledgebase/collections/{c}/files/{filename:path}` | 删除该文件全部图谱数据 |
| `POST /api/v1/knowledgebase/maintenance/clear-legacy-workspace` | 清空历史遗留空 workspace（升级前散落在工作目录根下的数据） |

### 2.6 三条硬约束（设计必须绕开或如实反映）

**① 后端当前没有任何前端托管能力。**
全项目 `app/` 下 `CORSMiddleware`、`add_middleware`、`StaticFiles`、`mount(` **零命中**。前端若跑在别的端口，一个请求都发不出去。

**② 没有会话列表接口。**
全项目无 `/sessions`、`/conversations`、`list_sessions`。后端只持有短期记忆，**会话历史只能由前端持久化**。

**③ 批量上传没有进度查询接口。**
`upload-bulk` 返回 202 + `task_id`，但五个 router 里**没有**任何 `/tasks/{id}` 之类的状态查询端点。
→ 用了它就只能提示"已提交，稍后刷新"，**无法显示进度**。

### 2.7 接口路径中的拼写陷阱

同一个文件里两个上传接口拼法不同，**且两个都真实存在**：

```
POST /api/v1/documents/kownledgebase/upload        ← 单文件，同步（沿用文件名 kownledgebase.py 的拼写）
POST /api/v1/documents/knowledgebase/upload-bulk   ← 多文件，202（正确拼法）
```

前端**必须写死各自正确的路径**，不可"顺手统一"——统一即 404。

---

## 3. 技术选型

**选型原则：全部用"Agent 网页"的主流稳定方案。**

项目作者**首次开发网页端**，因此这里刻意不引入任何小众或需要自行拼装的方案。
下面每个选择都是被大量项目验证过的默认答案——**遇到问题时搜得到答案，比"更先进"重要得多**。

### 3.1 依赖清单与版本下限

版本为 **2026-09-19 实查**（走 `registry.npmmirror.com` 镜像，非凭记忆）：

| 项 | 包 | 版本下限 | 说明 |
|---|---|---|---|
| 构建 | `vite` | ≥ **8.3.0** | 用户指定 |
| 框架 | `react` / `react-dom` | ≥ **19.3.0** | 用户指定 |
| 语言 | `typescript` | 随 Vite 模板 | 配合后端 Pydantic 模型做类型 |
| 样式 | `tailwindcss` | ≥ **4.3.3** | 用户指定 |
| 组件 | `shadcn`（CLI） | ≥ **4.21.0** | 组件源码复制进仓库，不锁定版本；聊天部分自己搭以贴合本项目**非标准** SSE |
| 服务端状态 | `@tanstack/react-query` | ≥ **5.103.1** | 数据几乎全是服务端状态（集合、文件、待审批），缓存与失效策略现成 |
| 路由 | `react-router-dom` | ≥ **7.18.4** | 多页面 |
| Markdown | `react-markdown` | 最新稳定 | Agent 回答常含表格 / SQL / 长路径 |
| 测试 | `vitest` | ≥ **5.0.1** | 与 Vite 同源；后端 Python 测试不受影响 |
| 测试 | `@testing-library/react` | 最新稳定 | 组件行为测试 |

### 3.2 三个必须记住的陷阱

**① Tailwind 是 v4，不是 v3——`tailwind.config.js` 已不存在。**

v4 的配置改为 CSS-first：入口 CSS 里 `@import "tailwindcss";` 加 `@theme { … }`。
按 v3 的写法去建 `tailwind.config.js`，**不会报错，但样式整片不生效**。
这是本次最容易踩、最难排查的坑（没有错误信息，只是"Tailwind 好像没起作用"）。

**② React 19 + react-router 7 + TanStack Query 5 是一套兼容组合，不要单独降级其中任何一个。**

**③ react-router v7 起，主包是 `react-router`；`react-router-dom` 仍作为兼容入口可用。**

计划里**统一用一个**，不混着 import（混用会导致同一份 router 上下文出现两个实例，
表现为"导航不生效"这类难查的怪问题）。

### 3.3 版本号由官方脚手架解析

实际安装用 `npm create vite@latest` 与 `npx shadcn@latest init`，
由它们解析出互相兼容的一组版本。上表是**下限**，不是硬钉值——
**除非真的遇到兼容问题，不要手工指定精确版本号**，那反而会把兼容组合拆散。

**明确不引入：**
- **Redux / Zustand** —— 真·客户端状态只有"当前会话 id"与"会话列表"，用 `useState` + Context + localStorage 足够。引入全局状态库是 YAGNI。
- **Assistant UI 等 AI 聊天组件库** —— 它们围绕 Vercel AI SDK 数据流协议设计，与本项目的自定义 SSE 不兼容，需要写适配层还要受其抽象约束。
- **任何 UI 框架（MUI/AntD）** —— 与 Tailwind + shadcn 路线冲突。

---

## 4. 架构

### 4.1 部署形态：后端挂载 dist（单源）

前端构建产物由 FastAPI 挂载，**前后端同源**，因此**不需要任何 CORS 配置**。

- 新增 `web/` 目录（与 `app/` 平级）
- `app/main.py` 增加静态挂载：服务 `web/dist/`，并对非 `/api/*` 路径回落到 `index.html`（SPA 路由）
- **开发期**：`vite.config.ts` 配置 `server.proxy` 把 `/api` 转发到后端端口，同样不需要 CORS

**这是本次唯一需要修改的后端代码**（约 5 行）。

### 4.2 目录结构

```
web/
├── package.json / vite.config.ts / tsconfig.json / index.html
├── src/
│   ├── main.tsx              入口
│   ├── App.tsx               路由表
│   ├── lib/
│   │   ├── api.ts            fetch 封装：基址、错误规整、JSON 解析
│   │   ├── sse.ts            自定义 SSE 解析器（读 ReadableStream，按 data: 切分）
│   │   └── utils.ts          cn() 等
│   ├── api/                  按后端资源分文件，只做请求与类型
│   │   ├── chat.ts           with_agent
│   │   ├── approvals.ts      pending / run 快照 / approval 决策
│   │   ├── documents.ts      RAG 集合与文件
│   │   └── knowledgebase.ts  图谱集合与文件
│   ├── components/
│   │   ├── ui/               shadcn 基元（Button / Dialog / Table / Toast …）
│   │   ├── layout/           AppShell / Sidebar
│   │   └── chat/             MessageList / Message / Composer /
│   │                         ApprovalCard / DegradedBanner / MessageMeta
│   └── features/
│       ├── chat/             ChatPage / useChatStream / sessions.ts
│       ├── documents/        DocumentsPage
│       └── knowledgebase/    KnowledgeBasePage
```

**分文件原则：** 按职责切，不按技术层切。`api/` 只放请求与类型，复杂逻辑放在 `features/` 内的 hook 里。

### 4.3 应用外壳

**单侧栏，内容随区切换：**

```
┌──────────────┬─────────────────────────────┐
│ 💬 对话       │                             │
│ 📄 文档       │        主区                 │
│ 🕸 知识库     │                             │
│ ──────────   │                             │
│ 下半栏：      │                             │
│  对话 → 会话列表                             │
│  文档 → RAG 集合列表                         │
│  知识库 → 图谱集合列表                        │
│ ──────────   │                             │
│ ⚠ 待审批 (N)  │  ← 常驻，不属于任何一个区      │
└──────────────┴─────────────────────────────┘
```

左栏顶部三个导航项固定；下半栏换成当前区最相关的列表；**底部"待审批"角标跨区常驻**。

**为什么选这个（而非顶部 Tab、或左栏只服务对话）：**
这是一个会持续加区的工具。左栏复用（内容随区切换）的代价是**一次性**的（每区定一次显示什么），
而"左栏只服务对话"的代价是**每次加区都付**（都要重新决定收不收左栏，导航位置还会跳动）。

---

## 5. 关键交互设计

### 5.1 对话区

**消息呈现：全宽文档式**（ChatGPT / Claude 式）——用户消息右对齐浅色块，助手回答全宽无气泡。

理由与项目实际输出形态有关：回答常含表格、SQL、`raw_data/sales_kb_docs/…` 这类长路径；气泡会把它压坏。
审批卡片还要展示工具名与参数预览，更需要横向空间。

**一次对话中可能出现的信息层次（自上而下）：**

1. **待审批卡片** —— 黄色边框，含工具名、参数预览（等宽字体）、批准/拒绝按钮、备注输入
2. **批准回执** —— 绿色卡片，含工具名、时间、审批意见
3. **降级警示条** —— 黄色横幅，仅当 `degraded=true`
4. **回答正文** —— 流式渲染，末尾带光标
5. **元数据脚注** —— 小字：`执行 N 步 · trace … · session …`

**降级处理：差异化。** 降级是**异常路径**，配得上显式提示；成功是常态，只留脚注不打扰。

⚠️ 这一条不是装饰需求。`degraded=true` 的语义是"Agent 按现有信息给了诚实的部分答案"——
上一个变更刚修好"重规划被排除时必须给出诚实部分答案"这条行为。**若该语义最后藏在一行灰字里，等于白修。**

**`strategy` 本轮不暴露给用户**，固定用后端默认值 `auto`。
理由：暴露它意味着要解释 `react` / `plan_execute` 的区别与各自代价，属另一件事的范围；
且默认值已是产品选定的路由策略。（后续若确有需要，加一个"高级设置"即可，不阻塞现在。）

### 5.2 SSE 状态机（`useChatStream`）

这是前端最需要测试的逻辑。状态：`idle → streaming → (awaiting_approval ⇄ streaming) → done | error`

要点：
- 用 `fetch` + `ReadableStream` 读 SSE（**不用 `EventSource`**——它是 GET-only，而我们的接口是 POST）
- 逐事件处理：`content` 追加到当前助手消息；`done` 收尾并记录元数据；`error` 标记失败
- 收到 `awaiting_approval`：**当前轮结束但对话未完成**，记录 `run_id` 与 `approvals`，保持消息流位置，插入审批卡片
- 用户批准/拒绝 → `POST /agent/runs/{run_id}/approval` → **继续往同一条助手消息追加** → 可能再次 `awaiting_approval`
- 中断（用户切页 / 组件卸载）：中止 fetch，但**不丢弃已渲染内容**——消息内容存在会话记录里（见 5.3），
  切回该会话即可恢复；正在流式中的那一轮标记为**"已中断"**，不假装它正常完成了

### 5.3 会话管理（前端持久化）

后端无会话列表接口，因此：

- 会话 id（`session_id`）由前端生成（`crypto.randomUUID()`）
- 会话列表（id / 标题 / 最后消息时间 / 消息记录 / **挂起的 `run_id`**）存 **localStorage**
- 标题取首条用户消息的**前 20 个字符**（超出加 `…`）；无用户消息时用"新会话"

**刷新恢复：** 页面加载时，若 localStorage 中某会话记录了挂起的 `run_id`，
调 `GET /agent/runs/{run_id}` 回查；若仍 `paused`，重建审批卡片。
若返回 `exists: false`（TTL 过期或后端降级为内存且已重启），标记该会话为"审批已失效"并给出说明。

### 5.4 待审批的跨区可见性

三层保障（**`/agent/approvals/pending` 已存在，无需改后端**）：

1. **内联** —— 审批卡片在对话流内（主路径）
2. **标签页角标** —— `document.title` 前缀 `(N) `
3. **左栏角标** —— 左栏底部常驻"⚠ 待审批 (N)"，点击跳回对应会话

第 3 层的数据来自 `GET /agent/approvals/pending`，**每 30 秒轮询一次**；
窗口重新获得焦点时立即刷一次（切走再切回来不必等 30 秒）。它同时是**唯一的兜底**：
没有它，"切到别的区就能看见挂起项"不成立。

**如实标注存储后端：** 接口返回的 `backend` 字段若为 `memory`，界面需提示"后端重启后将失效"。

### 5.5 文档区（RAG）

**以"文档"为中心。**

- 左栏：RAG 集合列表
- 主区：集合描述 + 文件表（文件名 / 切片数 / 删除）+ 上传按钮
- 上传：`POST /documents/upload`，**同步**，请求内返回结果 → 可直接显示成败与切片数
- 上传时可填 `description`（会 upsert 进集合注册表，**影响后续意图路由**，界面需说明）

### 5.6 知识库区（图谱）—— 独立定制

**以"图谱建得怎么样"为中心**，不与文档区共用组件。

理由（用户决策）：知识图谱后续要单独加功能，与文档区绑定会导致改一处牵动另一处。

- 左栏：图谱集合列表
- 主区：集合名 + 描述 + **图谱统计（实体数 / 关系数 / 文件数）** + 处理状态表 + 上传 / 维护入口
- 上传：用**单文件同步接口** `POST /documents/kownledgebase/upload`

**关于批量上传的取舍（重要）：**

`upload-bulk` 是 202 后台任务，而**后端没有任务状态查询接口**。因此：
- 若用 `upload-bulk` → 只能提示"已提交，稍后刷新"，**无法显示进度**
- 若用单文件同步接口 → 有确定成败，但大文件会让请求等很久

**本设计选择：主路径用单文件同步接口**，并在界面上如实说明图谱抽取耗时较长。
`upload-bulk` 作为后续可选项（需要后端先补任务查询接口）。

⚠️ 待你审阅时确认：是否接受"不做批量上传"。

---

## 6. 错误与边界处理

| 情况 | 处理 |
|---|---|
| SSE 中途断开 | 保留已渲染内容，标记"连接中断"，提供重试；不静默丢弃 |
| `{"error": …}` 事件 | 在当前消息位置显示错误，不覆盖已有内容 |
| 审批恢复返回 `run_not_found` | 明确提示"该审批已失效（后端重启或超时）"，并移除卡片 |
| 列表接口 500 | 用 Toast 报错 + 该区域显示可重试的错误态；不白屏 |
| 文件名含中文 / 空格 / 路径分隔符 | **必须 URL 编码**（后端显式声明支持中文，需编码） |
| 后端未启动 / 端口不通 | 首屏显示明确的"后端未连接"提示与排查指引，而非空白页 |
| `degraded=true` | 警示横幅（见 5.1） |
| `backend=memory` | 提示挂起项在后端重启后失效 |
| 空集合 / 空文件列表 | 空态文案 + 引导操作（上传） |

---

## 7. 测试策略

**前端测试（Vitest + React Testing Library）：**

重点测**逻辑**，不测像素：

1. **SSE 解析器**（`lib/sse.ts`）—— 分片到达、一条事件被切成多个 chunk、多事件在一个 chunk 里、`\n\n` 边界、非法 JSON 不崩
2. **`useChatStream` 状态机—— 这是最高价值的一组测试：**
   - 正常流：`content` 累积 → `done` 收尾，元数据正确
   - **审批循环**：`awaiting_approval` → 批准 → 续流 → **再次** `awaiting_approval`（同 `run_id`）
   - 拒绝路径：`approved: false` 带 `comment`
   - `degraded=true` 时降级标记被设置
   - 错误事件不破坏已有内容
3. **会话持久化** —— 写入 / 读回 / 挂起 `run_id` 的恢复与失效分支
4. **接口路径常量** —— 断言 `kownledgebase`（单文件）与 `knowledgebase`（批量）两条路径**各自正确**（防止"顺手统一"回归）

**后端测试：** 静态挂载改动需保证既有 376 项测试全部通过。

**手工验收：** 后端一条命令起来 → 浏览器访问 → 完成一次含审批与降级的真实对话。

---

## 8. 明确不在本次范围

| 项 | 原因 |
|---|---|
| **执行过程可视化**（展示规划 / 工具调用步骤） | `GET /agent/runs/{run_id}` 不返回完整步骤，需后端新增接口。用户已决定本期不做 |
| **审批中心独立页面** | 已用"左栏角标 + 跳回会话"覆盖；独立页面是后续可选项 |
| **登录 / 鉴权 / 多用户** | 后端无任何认证中间件，不在可做范围内 |
| **问题实体抽取** | 用户明确暂缓（另一条线） |
| **批量上传与进度显示** | 后端无任务查询接口（见 5.6） |
| **后端 `retrieval_hint` 相关改动** | 属另一个变更的范围 |

---

## 9. 交付与运行

**开发：**
```bash
# 终端 1：后端
uvicorn app.main:app --reload

# 终端 2：前端（Vite dev server，/api 代理到后端）
cd web && npm run dev
```

**日常使用（单进程）：**
```bash
cd web && npm run build          # 产出 web/dist/
uvicorn app.main:app            # 后端挂载 dist/，同源访问
```

**静态挂载的具体约定（写死，避免实现时再猜）：**

- 前端路由：`/` → 对话区，`/documents` → 文档区，`/knowledgebase` → 知识库区，`:sessionId` 走查询参数
- **静态资源**：挂载 `web/dist/assets/*` 等真实文件
- **SPA 回退**：凡**不以 `/api/` 开头**且不是已存在的静态文件的路径，一律返回 `web/dist/index.html`
  ——否则 `/documents` 这类前端路由一刷新就 404
- `/api/v1/*` **一律**交给现有 router，静态挂载**不得**拦截
- `web/dist` 目录不存在时（前端尚未构建）**跳过挂载并打一条 WARNING**，后端照常启动
  ——避免"没构建就跑不起后端"

⚠️ 若将来 `api_prefix`（当前 `/api/v1`）改为 `/`，上述"以 `/api/` 开头"的判定需同步调整。

---

## 10. 已确认的决定（审阅记录 2026-09-19）

以下三项经项目作者审阅确认，**不再是待议项**，实现时直接按此执行：

1. **知识库区不做批量上传**（第 5.6 节）——**已接受**。
   理由：后端无任务进度查询接口，`upload-bulk` 的 202 只能"提交后刷新"、无法显示进度。
   主路径改用单文件同步接口 `POST /documents/kownledgebase/upload`。

2. **修改 `app/main.py` 挂载 `web/dist`**（第 4.1 节）——**已确认**。
   这是本设计**唯一**需要改动的后端代码，约 5 行。

3. **开发与日常两种运行模式并存**（第 9 节）——**已认可**。
   开发用 Vite dev proxy；日常用后端托管 `dist`（单进程、同源、无 CORS）。

**附加确认（技术选型原则）：** 项目作者**首次开发网页端**，明确要求
"按 Agent 网页通用的来、用稳定的"。第 3 节即该原则的落实，
并在 3.1 记录了实查的版本下限、3.2 记录了三个具体陷阱。

**同批确认的五处规格收紧**（见第 11 节自审记录）：静态挂载约定写死、
待审批轮询 30 秒、中断内容存储位置、"已中断"标记、`strategy` 不暴露、标题截断 20 字符。

---

## 11. 规格自审记录

写完本规格后按四步自审（占位符扫描 / 内部一致性 / 范围 / 歧义），
发现并修正 5 处，均已由作者确认：

| # | 问题 | 性质 | 修正 |
|---|---|---|---|
| 1 | 第 9 节把静态挂载"留到实现时确认" | **占位符** | 写死：`/assets/*` 走静态；**非 `/api/` 前缀**回退 `index.html`；`dist` 不存在则跳过挂载并打 WARNING |
| 2 | 待审批轮询未给间隔 | 歧义 | 每 **30 秒**一次；窗口重新获得焦点时立即刷一次 |
| 3 | 第 5.2 节"中断不丢弃已渲染内容"未说内容存哪 | 歧义 | 明确存在会话记录（第 5.3 节），切回即恢复；流式中断的那一轮标记 **"已中断"**，不假装完成 |
| 4 | `strategy` 参数未表态 | 歧义 | **不暴露**，固定用 `auto`（暴露它需解释 `react`/`plan_execute` 的差别与代价，属另一件事） |
| 5 | 会话标题"截断"未给长度 | 歧义 | 前 **20 个字符**，超出加 `…`；无用户消息时用"新会话" |

**范围结论：** 三个区虽属不同后端资源，但共用同一套外壳、数据层与 SSE 机制，
且作者已明确选择"一次做"，因此**维持单个规格、单个实现计划**，不拆分。
代价是计划较长，但拆成三份会产生三份重复的外壳与数据层任务。
