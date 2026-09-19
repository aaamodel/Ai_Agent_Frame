## 1. 依赖与前置（先确认可行性，再动代码）

- [x] 1.1 安装 `vanna` / `chromadb` 并写入 `requirements.txt`；验证：`python -c "import vanna, chromadb"` 成功，且记录安装耗时（chromadb 体量较大）
- [x] 1.2 确认硬编码通道可用：`DASHSCOPE_API_KEY` 可读、DashScope OpenAI 兼容端点可达、ChromaDB 目录可写；验证：用参照实现的配置跑通一次最简 `generate_sql`
- [x] 1.3 若 chromadb 在本机不可用，记录退路（改用 Vanna 的其他向量存储实现）并**暂停**等待确认——不得静默换方案

## 2. 建库与数据迁移（数据先行，可独立验证）

- [x] 2.1 按字段字典为 **7 张业务表**写 `CREATE TABLE`：主键、类型、`NOT NULL`、枚举 `CHECK`，中文列名与源 Excel 表头逐字一致；验证：建表后 `PRAGMA table_info` 的列名与源表头一一对应，且 `sqlite_master` 中能看到 `CHECK`
- [x] 2.2 从 `raw_data/sales_intel` 的 7 张业务 sheet 导入（先建表后 `append`）；验证：每张表行数等于源 sheet 数据行数（45/34/12/12/12/6/7）
- [x] 2.3 导入即校验约束：任一行违反主键/`CHECK`/`NOT NULL` MUST 整体失败而非部分成功；验证：单测用一行越界枚举值断言导入失败且报错指出合法取值
- [x] 2.4 业务表扩量至 **4 倍**（128 → 512 行）：只使用既有枚举与维度组合，关联字段取自扩量后的线索表；验证：① 每表行数为扩量前 4 倍；② 全表约束校验通过；③ 无孤儿引用（输赢单的线索编号均能在线索表找到）
- [x] 2.5 提供"从库导出 xlsx"的脚本（xlsx 定位降级为导出视图）；验证：导出文件能被 Excel 打开且行列与库一致

## 3. 规则与口径的分流（**不是** 8 张 sheet 全量进 RAG）

> 见 design D7 / D7b：列名/类型/枚举已被 DDL 覆盖，**不进 RAG**；
> 真正需要单独处理的是"口径"与"纯规则"两类。

- [x] 3.1 把**纯规则**文本化：阶段流转规则 / 折扣权限 / 组合策略；验证：三份文档可读且完整保留原文判据（进入标准、退出标准、折扣下限、审批人）
- [x] 3.2 从 5 张字段字典中**只提取口径与判据**（如"员工规模 ≥ 200 为 ICP 达标""写入前需查重"），**丢弃**列名/类型/枚举取值（DDL 已覆盖）；验证：提取结果中不含纯 schema 复述，且每条口径都能追溯到来源 sheet
- [x] 3.3 把口径训练进 Vanna 的 `documentation`（脚本入口，见 D7b）；验证：提一个**依赖口径**的问题（如"ICP 达标的线索有多少"），生成的 SQL 中 MUST 含 `员工规模 >= 200` 这类判据，而非让模型自行假设
- [ ] 3.4 三份规则文档导入既有 `sales_kb` 集合（不新建集合）；验证：`rag_knowledge_search` 能检索到阶段流转规则与折扣权限原文
- [x] 3.5 确认业务库中**不存在**这些规则表；验证：业务库表清单仅含 7 张业务表

## 4. SQL 工具实现（Vanna 核心，注册为 BaseTool 子类）

- [x] 4.1 实现 Vanna 封装（继承 `ChromaDB_VectorStore` + `OpenAI_Chat`，**去掉 pg 分支与跨库路由**）；验证：单测断言只建立一个 SQLite 连接，且不存在任何 postgres 相关代码路径
- [x] 4.2 实现 `sales_sql_query`（自然语言 → 结果，返回中附生成的 SQL）；验证：单测断言返回值含结果与 SQL 两部分，且 SQL 引用的表/列在库中真实存在
- [x] 4.3 只读护栏：查询通道拒绝 `INSERT`/`UPDATE`/`DELETE`/`DROP`/`ALTER`/`ATTACH` 等；验证：单测逐类断言被拒绝且给出"只读"说明
- [x] 4.4 失败可辨识：生成失败/执行报错必须返回可读错误，"查到 0 行"必须区别于"执行失败"；验证：单测覆盖三种返回形态
- [x] 4.5 实现 `sales_sql_write`：语义参数 → 参数化 `UPDATE`，列名走 `PRAGMA table_info` 白名单、值用 `?` 占位；验证：单测断言列名不在白名单时被拒、值含引号时不被注入
- [x] 4.6 写入护栏：`rowcount == 0` 拒绝、`rowcount > 1` 拒绝并列出行；验证：单测覆盖唯一命中/0 命中/多命中三种情形
- [x] 4.7 写工具登记进 `config.py` 危险工具名单；验证：配置读取中包含该工具名
- [x] 4.8 DDL 增量同步：新增/变更/删除表后同步训练数据；验证：单测断言新建表后可被查询、删表后不再被引用
- [x] 4.9 硬编码配置缺失时明确报错（不静默降级）；验证：单测在缺 Key / 端点不可达时断言返回明确初始化失败信息

## 5. 契约切换（路由全部指向新工具；此时新旧并存，可回退）

- [x] 5.1 意图树工具绑定与路由文案改用 SQL 工具（8 处）；验证：意图树中不再出现 `local_excel_*`
- [x] 5.2 `SKILL.md` 的 allowed-tools 与数据资产地图改为指向库与 SQL 工具（16 处）；验证：文件中不再出现 `local_excel_*`
- [x] 5.3 `planner.py` / `summarize_node.py` 提示词改造：取数引导改为 SQL 工具，写操作判定改为 SQL 写工具；验证：单测断言写操作判定覆盖 SQL 写工具
- [x] 5.4 `init_tools.py` 注册新工具、`config.py` 危险名单同步；验证：工具清单中同时含新工具且不含已退役工具
- [x] 5.5 清理其余工具文案中对 Excel 工具的指引（`filesystem_tools.py`、`sales_report.py`、`asset_map.py`）；验证：全仓扫描无 `local_excel_*`
- [x] 5.6 evals 同步：`_tool_schemas.json`、`export_tool_schemas.py`、`test_eval_gate.py`、`test_metrics.py`、`test_run_tool.py`；验证：evals 可运行且断言不引用已退役工具
- [x] 5.7 受影响测试改造：`test_excel_affordance.py`、`test_skill_asset_map.py`、`test_step_correction_candidates.py`、`test_agent_goal_injection.py`；验证：`pytest 测试 -q` 全绿
- [x] 5.8 文档同步：`README.md`、`rag_data/other/销售助手评测问题集.md`、`rag_data/市场活动效果.md`；验证：文档中不再引导按文件路径取数

## 6. 删除旧工具（最后一步，避免中间态不可用）

- [x] 6.1 删除 `app/core/tools/builtin/localexcel.py` 与 `excel_query.py` 及其注册；验证：文件不存在，`python -c "import app.core.tools.builtin.localexcel"` 失败
- [x] 6.2 全仓扫描确认无悬空引用（代码、技能文档、评测集、测试，openspec 归档记录除外）；验证：扫描结果为 0 处
- [x] 6.3 全量回归；验证：`pytest evals 测试 benchmark -q` 通过，且与变更前基线一致（当前存在 1 个既有的 cost gate 失败项，非本变更引入，不计入）

## 7. 端到端验证

- [x] 7.1 自然语言取数端到端：用"三季度哪条产品线赢单金额最高"类问题实跑；验证：一次调用拿到结果，且生成的 SQL 表/列真实存在
- [x] 7.2 受约束写入端到端：改某条线索的负责人；验证：唯一命中成功、0 命中与多命中均被拒，且危险工具审批流程生效
- [x] 7.3 派生指标实时计算：查询季度活动投入/赢单汇总；验证：结果来自对明细表的聚合，库中不存在汇总表
- [x] 7.4 **遗留记录**：既有主 spec `agent/excel-query` 描述的已是本次删除的工具，归档前需另开变更以 MODIFIED/REMOVED 处理；验证：在变更制品中明确记载该遗留（本任务为记录项，不改该 spec）
