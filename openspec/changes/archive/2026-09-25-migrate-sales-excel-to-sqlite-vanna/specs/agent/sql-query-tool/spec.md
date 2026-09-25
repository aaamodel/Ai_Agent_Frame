## Purpose

让执行期用一句自然语言直接向业务库要答案：由工具负责生成并执行 SQL，模型不再需要拼装
"文件路径 + 工作表名 + 过滤列 + 分组列 + 聚合列 + 聚合函数"这类参数组合。核心逻辑使用
Vanna（DDL/问答对检索 + LLM 生成 SQL），但以本项目的 `BaseTool` 子类形式注册。

## ADDED Requirements

### Requirement: 自然语言问题必须能直接转成 SQL 并执行

系统 MUST 提供执行期 SQL 工具，接受一句自然语言业务问题，返回结构化查询结果。

核心逻辑 MUST 使用 Vanna（或其等价的 schema 检索 + 生成式 SQL 方案）：
MUST 基于**库中实际存在的表结构**生成 SQL，MUST NOT 让模型凭记忆臆造表名与列名。

工具 MUST 注册为本项目的 `BaseTool` 子类，MUST NOT 注册为 langchain 伪工具装饰器形式。

#### Scenario: 一句中文问题得到结果

- **WHEN** 传入"三季度哪条产品线赢单金额最高"这类问题
- **THEN** 系统 MUST 返回对应的结构化结果，且 SQL 中引用的表名与列名 MUST 在库中真实存在

#### Scenario: 结果为空与失败必须可区分

- **WHEN** 查询执行成功但无匹配行
- **THEN** 结果 MUST 明确表达"查询成功但无数据"，MUST NOT 与"执行失败"或"数据不存在"混淆

#### Scenario: SQL 生成或执行失败必须可读

- **WHEN** SQL 生成失败或执行报错
- **THEN** 系统 MUST 返回包含错误原因的可读信息，MUST NOT 静默返回空结果

### Requirement: 只读查询必须拒绝非只读语句

查询通道 MUST 只执行只读语句（`SELECT` / `WITH ... SELECT`）。

传入写入或结构变更语句（如 `INSERT` / `UPDATE` / `DELETE` / `DROP` / `ALTER` /
`ATTACH` / `PRAGMA` 写操作）时，系统 MUST 拒绝执行并给出明确错误，MUST NOT 静默改写。

#### Scenario: 只读查询正常执行

- **WHEN** 执行一条 `SELECT` 查询
- **THEN** 系统 MUST 返回结果

#### Scenario: 写入语句被拒绝

- **WHEN** 向查询通道传入 `UPDATE` / `DELETE` / `DROP` 等语句
- **THEN** 系统 MUST 拒绝执行，且 MUST 明确说明查询通道只读

### Requirement: 写入必须受约束，并沿用既有的命中数护栏

系统 MUST 提供受约束的写入通道，把"条件列 = 值 → 改某列"的语义更新映射为
`UPDATE ... SET ... WHERE ...`。

- 条件命中 0 行时，系统 MUST 拒绝写入并说明未命中（对应 `rowcount == 0`）；
- 条件命中多行时，系统 MUST 拒绝写入并列出命中行（对应 `rowcount > 1`），
  防止一次误改一批；
- 写入 MUST 经过既有的危险工具审批流程，MUST NOT 绕过。

理由：这些护栏在现有 Excel 写工具里已被实测验证是必要的（写错字段、误改多条都是真实
事故）。换到 SQL 后语义完全等价，护栏 MUST NOT 随存储更换而丢失。

#### Scenario: 条件唯一命中时写入成功

- **WHEN** 更新条件唯一命中一行
- **THEN** 系统 MUST 完成更新，并返回被改字段的旧值与新值

#### Scenario: 条件命中 0 行时拒绝

- **WHEN** 更新条件未命中任何行
- **THEN** 系统 MUST 拒绝写入，且 MUST NOT 静默成功

#### Scenario: 条件命中多行时拒绝

- **WHEN** 更新条件命中多行
- **THEN** 系统 MUST 拒绝写入，并 MUST 列出命中行供调用方收紧条件

### Requirement: 生成 SQL 所依据的 schema 必须与实际库结构同步

生成 SQL 所依赖的 schema 知识 MUST 来自**当前库中实际的表结构**，MUST 在表结构变化时
可增量同步，MUST NOT 依赖硬编码的表名清单。

同步 MUST 是增量的：新增/变更/删除的表 MUST 被相应地加入/更新/移除。

#### Scenario: 新建表后可被查询到

- **WHEN** 业务库中新增了一张表并完成同步
- **THEN** 后续的自然语言查询 MUST 能够引用该表

#### Scenario: 表结构变更后同步更新

- **WHEN** 某张表新增或删除了列并完成同步
- **THEN** 后续生成的 SQL MUST 依据变更后的列结构

#### Scenario: 已删除的表不再被引用

- **WHEN** 业务库中某张表被删除并完成同步
- **THEN** 后续生成的 SQL MUST NOT 引用该表

### Requirement: 本工具只支持 SQLite，且模型调用不走项目模型路由

系统 MUST 只连接 SQLite，MUST NOT 引入 PostgreSQL 或其他引擎的分支与跨库路由。

本工具的 LLM 与向量模型 MUST 使用**硬编码配置**（DashScope OpenAI 兼容端点 + 指定模型 +
ChromaDB 本地向量库），MUST NOT 接入项目的 ModelRouter。

⚠️ 这是**有意为之的例外**，代价 MUST 被显式接受：该通道不受多厂商方言适配、
熔断降级、调用预算与 Langfuse 追踪覆盖。

#### Scenario: 只连接 SQLite

- **WHEN** 工具初始化
- **THEN** MUST 建立且仅建立一个 SQLite 连接，MUST NOT 尝试连接其他数据库类型

#### Scenario: 跨库方言被拒绝

- **WHEN** 生成或执行的 SQL 含非 SQLite 方言、或试图跨库 `JOIN`
- **THEN** 系统 MUST 拒绝并明确说明只支持 SQLite 单库

#### Scenario: 硬编码配置缺失时明确报错

- **WHEN** 硬编码的 API Key / 端点 / 向量库路径不可用
- **THEN** 工具 MUST 返回明确的初始化失败信息，MUST NOT 静默降级为不可用状态
