## Why

Excel 取数工具反复踩坑，**根因不是工具写得不好，而是 Excel 本身不携带 schema**。

面对单元格，模型只能猜"这一列是产品线还是公司全称"，于是要补偿：参数越加越多
（`local_excel_read_tool` 已 9 个参数）、列名靠 `difflib` 模糊匹配、预览只有几行导致
"预览里没看到 = 数据不存在"的误判（2026-08 销售复盘事故里为此空转 10+ 轮）。

换到 SQLite 后，`CREATE TABLE` 把 schema 定死——`PRIMARY KEY` / `CHECK` / `NOT NULL`
就是模型看得见的确定约束；`LIKE` 天然解决模糊匹配，`GROUP BY` 天然解决聚合，`JOIN`
天然解决多表关联。之前在 Excel 上想补的那些算子，SQL 全都有且是标准。

## What Changes

- **新增**：业务数据的权威存储从 Excel 改为 **SQLite**（单文件，无需起服务）。
  建表带显式约束（主键 / 枚举 CHECK / NOT NULL），由 `raw_data/sales_intel` 的
  **业务类 sheet** 导入。
- **新增**：执行期 SQL 工具，以 Vanna 为核心（RAG 式 DDL/问答对检索 + LLM 生成 SQL）。
  适配为本项目 `BaseTool` 子类（**不是** langchain 伪工具）。只支持 **SQLite**（参照实现里
  的 PostgreSQL 分支与多引擎路由**不引入**）。LLM 与向量模型**不走项目 ModelRouter**，
  按参照文件硬编码（DashScope OpenAI 兼容端点 + `qwen3.7-plus-2026-05-26` + ChromaDB）。
  **BREAKING**：这是全项目唯一不走 ModelRouter 的模型调用，熔断降级/多厂商方言/
  Langfuse 追踪均不覆盖该通道。
- **新增**：SQL **写**工具，把语义更新映射为 `UPDATE ... WHERE ...`，
  沿用现有"0 命中拒绝 / 多命中拒绝"的护栏（对应 `cursor.rowcount`）。
- **移除（BREAKING）**：`app/core/tools/builtin/localexcel.py`、`excel_query.py`，
  以及 `local_excel_read_tool` / `local_excel_write_tool` / `local_excel_query_tool`
  三个工具的全部注册与引用。
- **契约全量改造**：`local_excel_*` 当前被约 **30 个文件**引用——意图树工具绑定
  （8 处）、`SKILL.md`（16 处，含数据资产地图）、`planner` / `summarize_node` 提示词、
  `config.py` 危险工具名单、`init_tools.py` 注册、evals 黄金集与 schema、
  以及 4 个测试文件。本次**全部同步，不留悬空引用**。
- **数据分类（本次决策）**：
  - 业务数据（7 张表，共 128 行）→ SQLite；
  - 规则/元数据（5 张字段字典 + 阶段流转规则 + 折扣权限 + 组合策略）→ **已有 RAG 知识库**，
    不走 SQL；
  - 计算派生（季度汇总 / 区域产品透视 / 原因分类）→ **不落库**，由 SQL 实时算。
- **数据扩量**：业务表行数**变为 4 倍**（净增 3 倍），128 行 → **512 行**。
- **保留**：`raw_data/sales_intel/*.xlsx` 全部保留，定位降级为**导出视图**，不再承担查询职责。

## Capabilities

### New Capabilities

- `platform/sales-sqlite-store`: 销售业务数据的权威存储——建库与建表（含约束）、
  从 Excel 导入、扩量口径，以及"哪些 sheet 进库 / 哪些进 RAG / 哪些根本不存"的分类契约。
- `agent/sql-query-tool`: 执行期 SQL 工具——自然语言生成 SQL、只读查询执行、
  受约束写入，以及"不走 ModelRouter、只支持 SQLite"的边界。
- `agent/excel-tool-retirement`: Excel 取数工具退役后的一致性契约——系统不得再暴露或
  引用已退役工具，取数路径必须指向 SQL 工具。

### Modified Capabilities

<!-- 无。既有 8 个能力均不覆盖业务数据存储与 Excel 工具存废。
     注意：`agent/excel-query`（表格概览须披露可聚合能力）描述的是**已退役工具**的
     行为，本变更删除该工具后其需求即失效——但删除能力需走 MODIFIED/REMOVED，
     超出本次范围，故作为已知遗留列在 design.md 的 Risks。 -->

## Impact

**依赖**

- 新增 `vanna`、`chromadb`（**当前均未安装**）。`chromadb` 体量较大，首次安装耗时明显。
- ChromaDB 需要可写目录存放向量（`./chroma_db` 或配置路径），需确认部署环境可写。

**代码**

- 新增：`app/core/tools/builtin/sql_vanna.py`（BaseTool 子类）、建库/建表/导入脚本
- 删除：`app/core/tools/builtin/localexcel.py`、`app/core/tools/builtin/excel_query.py`
- 改造：工具注册、意图树工具绑定、`SKILL.md`（允许工具 + 数据资产地图）、
  `planner.py` / `summarize_node.py` 提示词、`config.py` 危险工具名单、
  `filesystem_tools.py` / `sales_report.py` 中指向 Excel 工具的文案、
  `app/core/skill/asset_map.py`
- 评测与测试：`evals/_tool_schemas.json`、`evals/tools/export_tool_schemas.py`、
  `evals/test_eval_gate.py`、`evals/test_metrics.py`、`evals/test_run_tool.py`、
  `测试/test_excel_affordance.py`、`测试/test_skill_asset_map.py`、
  `测试/test_step_correction_candidates.py`、`测试/test_agent_goal_injection.py`
- 文档：`README.md`、`rag_data/other/销售助手评测问题集.md`、`rag_data/市场活动效果.md`

**数据与运行**

- 新增 SQLite 库文件（业务表，512 行量级，体积 KB 级）
- 规则类 sheet 需导入 RAG 知识库（新增集合或并入 `sales_kb`，见 design 决策）
- SQL 工具每次调用会产生**额外的大模型调用**（生成 SQL + 可能的自修复），
  且**不经 ModelRouter**——不受现有调用预算与熔断保护，需评估注入预算账本的方式

**不在本次范围**

- PostgreSQL 支持与多引擎路由（参照实现里的 pg 分支**不引入**）
- Excel 文件的删除（本次保留为导出视图）
- 为业务员提供改数据的前端（进库后 Excel 编辑与库不同步的问题，见 design Risks）
- 既有 `agent/excel-query` 能力的删除（需 MODIFIED/REMOVED delta，留作后续）
