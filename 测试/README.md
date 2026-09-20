# 测试目录按变更提案分类

每个子目录对应**一个变更提案**（openspec change / superpowers plan），
目录名优先用提案名，没有提案的用最能表达该变更的映射名。

规则：**新增测试放进产生它的那个提案的目录**，不要再堆在 `测试/` 根下。
`pytest 测试/` 会递归收集，命令行不用改。

## 映射表

| 目录 | 提案依据 | 覆盖内容 |
|---|---|---|
| `llm-token-streaming/` | superpowers plan `2026-09-20-llm-token-streaming` | LLM 输出逐字流式：旁观通道、增量 JSON 抽取、显示层分流、调用器流式与探测回落、SSE 接线 |
| `agent-goal/` | `openspec/specs/agent/agent-goal` | 本轮目标抽取与注入 |
| `plan-execute-control/` | `openspec/specs/agent/plan-execute-control` | 计划执行控制协议、账本、子任务分配与覆盖、Planner 输入块 |
| `replan-and-step-correction/` | git `cc7648e`（重规划门控 + 步骤修正） | 显式步骤提示、重规划上下文与触发、步骤修正候选与证据缺口 |
| `llm-call-budget/` | `openspec/specs/platform/llm-call-budget` | 单次调用独立计时、重试预算、各 LLM 入口参数一致（含 LightRAG） |
| `excel-to-sqlite/` | `openspec/changes/migrate-sales-excel-to-sqlite-vanna` | 销售数据迁移到 SQLite/Vanna、SQL 工具、技能资产映射 |
| `kb-collection/` | `openspec/changes/archive/2026-09-18-add-collection-enum-and-drop-retrieval-hint` | 集合枚举与元数据 |
| `frontend-console/` | 前端控制台变更（git `0f64a59`） | 前端构建产物的静态托管与 SPA 回退 |
| `shared/` | 跨提案 | 全链路状态机回归等被多个提案共用的测试 |

## ⚠️ 归类是"最好努力"，不是客观还原

迁移时用 git 历史（哪个提交引入的）+ openspec 文档里点名的测试文件来定归属，
但**历史本身不是干净的 1:1 映射**：例如 `5c3d966` 一个提交就同时打包了
agent-goal、plan-execute-control、plan-ledger、图状态机四件事，
它们的测试只能按主题拆到不同目录——**这是我判断的结果，不是提交记录的直接体现**。

所以：

- 发现某个文件放错了，直接 `git mv 测试/<旧>/x.py 测试/<新>/x.py` 即可，
  目录名只是分类，不被任何代码引用
- 新增提案时**同时建目录**，别等以后再补——补的时候就得靠考古了
