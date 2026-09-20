## 1. 参数模型的约束表达能力（打底）

> 依据 design.md D2。当前 `ToolParameter` 无法表达 enum，数组连 `items` 都没有——
> 不先补这层，"给集合参数加枚举"无从下手。

- [x] 1.1 为 `ToolParameter` 增加可选 `enum` 与 `items` 字段，并让 `schema_parameters()` 在有值时合并进 property 定义；验证：单测断言 `enum` / `items` 出现在导出结果中，且**未设置这两个字段的参数导出结果与改动前逐字一致**
- [x] 1.2 确认既有工具（`rag_knowledge_search` / `knowledge_graph_search` / `local_excel_read_tool` 等）的 schema 导出无回归；验证：单测断言各工具导出结构不变，`pytest 测试 -q` 全绿

## 2. 集合枚举的来源与口径

> 依据 design.md D1 / D3 与 `agent/kb-collection-routing` 的前两条需求。
> 枚举必须**运行时**取自注册表，且只含有描述的集合。

- [x] 2.1 在 `KbCollectionRegistry` 上提供"具备非空描述的集合（名 + 描述）"查询；验证：单测断言描述为空的集合被排除、有描述的按注册顺序返回
- [x] 2.2 覆写 `rag_knowledge_search` 的 `schema_parameters()`，为 `collection_names` 注入动态枚举与逐条描述，并去掉参数说明里不存在的示例 `如 ['hr_docs']`；验证：单测断言 ① 枚举恰好等于"有描述集合"全集；② 每个枚举值都在说明里逐条出现；③ 说明文本中**不含任何注册表之外的名字**
- [x] 2.3 验证枚举随注册表刷新：`replace` 新增/删除集合后，下一次 `schema_parameters()` 立即反映变化；验证：单测断言无需重建工具实例即可看到变化

## 3. 越界取值的静默过滤与回落

> 依据 design.md D4 / D5 与 `agent/kb-collection-routing` 的第三条需求。
> 过滤点必须在 `rag_service` 的白名单入参处——那是所有调用路径的唯一收口。

- [x] 3.1 在 `rag_service.retrieve_contexts` 的集合白名单入参处按注册表过滤非法名；验证：单测断言越界名被丢弃、合法名保留
- [x] 3.2 实现两条回落路径：过滤后非空 → 仅在合法集合内检索；过滤后为空 → **不限定集合**检索（全库），不得失败；验证：单测分别覆盖两种输入
- [x] 3.3 为过滤与回落留下可观测记录（至少含被丢弃的取值）；验证：单测断言记录中能查到被丢弃的取值

## 4. retrieval_hint 全链路移除

> 依据 design.md D6 / D7 与 `platform/kb-collection-metadata`。这是 **BREAKING** 且
> `DROP COLUMN` **不可逆**，因此 4.1 必须先于 4.3 完成。

- [x] 4.1 **（部署前人工步骤）** 导出并留档现存映射：`SELECT name, retrieval_hint FROM vector_collections;`。这是回滚时唯一的数据来源；验证：导出文件已保存且行数与集合数一致
- [x] 4.2 移除 `agent_schemas` 的该字段、`models.py` 的列定义、`collection_repo.py` 的读写分支；验证：单测断言从存储构造描述符时不再读取该列
- [x] 4.3 移除 `document.py` / `kownledgebase.py` 中该字段的 Form 入参与响应字段；验证：单测断言 ① 接口签名不含该参数；② 响应不含该字段；③ 调用方多传该字段时**请求仍成功**（不 422）
- [x] 4.4 移除 `KbCollectionDescriptor.retrieval_hint` 字段与 `intent_text()` 中的拼接分支，路由语义只由 `description` 承担；验证：单测断言渲染文本只含一份语义文本、且不再出现"适用检索时机"字样
- [x] 4.5 在 `app/main.py` 既有轻量 DDL 块内追加幂等的 `DROP COLUMN IF EXISTS`；验证：① 在已有该列的库上启动一次，列消失；② **再次启动不报错**（`IF EXISTS` 生效）
- [x] 4.6 更新集合资产 `向量化数据库的集合.json`，移除该字段；验证：文件中不再出现该键，且其余字段与真实集合一一对应
- [x] 4.7 全仓扫描确认无残留引用（含 `evals/tools/reembed_from_db.py` 等文档字符串）；验证：`retrieval_hint` 仅出现在本变更的 openspec 制品与历史记录中，应用代码与资产内为零

## 5. 回归与端到端

- [x] 5.1 回归全部现有测试；验证：`pytest evals 测试 benchmark -q` 通过，且与变更前基线一致（当前存在 1 个既有的 cost gate 失败项，非本变更引入，不计入）
- [ ] 5.2 **意图匹配回归**：`intent_text()` 少一段文本会改变动态集合节点的匹配输入；验证：动态集合的意图命中用例通过，命中率不低于变更前（需要记录变更前基线数字）
- [x] 5.3 用本次 trace 的原始问题（"私有化部署 上线周期/实施费 收费标准"类）端到端实跑；验证：① 工具调用参数中的集合名**全部属于真实注册表**；② 不再因空召回触发硬熔断；③ 与变更前（2 个子任务全空召回 + 工具熔断 + 降级收尾）对比
- [x] 5.4 构造越界场景：桩掉模型输出一个不存在的集合名；验证：① 检索仍返回全库结果、请求不失败；② 记录中能查到被丢弃的取值
- [x] 5.5 观测单请求提示词预算增量：8 个集合的描述进入 Function Calling 定义后，确认增量与 `platform/llm-call-budget` 的既有约束相容；验证：记录变更前后单请求输入字符数并对比

---

## 6. 实现后补记（实测证据与两处前提修正）

### 6.1 任务 4.1 的前提不成立：该列从未存在，无数据可备

查 `information_schema.columns` 的实际结果：

```
vector_collections 实际列: name | description | created_at | updated_at | engine
```

**没有 `retrieval_hint` 列。** 原因是项目的已知弱点：模型新增了该列，而 `create_all`
不会 ALTER 已存在的表，`main.py` 的"轻量补列"只补过 `engine`。因此这条 DDL 在本部署里
是**空操作**，也没有任何数据处于风险中——原任务设想的"导出留档"没有对象。
DDL 仍然保留：在确实建过该列的部署上它才是必要的。

幂等性已实测：连续执行 3 次 `DROP COLUMN IF EXISTS retrieval_hint` 全部 OK，列集合不变。

### 6.2 顺带修复了一处写入路径故障（本变更的额外收益）

既然模型声明了表里不存在的列，那么改动前任何 `vector_collections` 写入都会失败。
用一个**独立映射**（在事务内试写后回滚，不改动应用代码）精确复现：

```
结果: 写入失败 -> ProgrammingError: asyncpg.exceptions.UndefinedColumnError:
      column "retrieval_hint" of relation "vector_collections" does not exist
探测行残留数: 0
```

也就是说：**改动前，上传文档时填的集合描述根本存不进库**——异常被上传接口宽泛的
`except Exception` 吞掉（回滚 + 一条 warning），集合描述永远登记不上。移除该列后，
同一次 upsert 试写**成功**。这是本变更没有预期到的收益。

### 6.3 端到端：注册表必须在"生产路径"下才非空

`app.main` 的 lifespan 会调 `refresh_kb_registry` 填充注册表；而 `EvalRuntime` **不走
lifespan**，直接跑时注册表是空的（枚举随之为空）。这不是缺陷，但意味着端到端验证必须先
按生产方式填充注册表，否则验证不到枚举。

按生产路径填充后（7 个集合），模型可见的 `collection_names` 枚举**恰好等于这 7 个真实
集合**，说明逐条 7 行。

用本次 trace 的原始问题（私有化部署的上线周期 / 实施费 / 收费标准）实跑：

| | 变更前 | 变更后 |
|---|---|---|
| 传入检索的集合名 | `['product_docs','sales_policies']`、`['pricing_guide','sales_policies']` | **`['sales_kb']`** |
| 越界取值 | 全部越界 | **无** |
| 结果 | 两个子任务空召回 → 工具硬熔断 → 整轮降级收尾 | `success=True`，正常作答 |
| 大模型调用 / 输入字符 | 13 次 / 约 37,400 | **4 次 / 10,222** |

修复后的行为正是设计意图：模型不再编造，而是从枚举里选出描述含"产品与定价"的
`sales_kb`。

单请求预算：同一问题在变更前一轮实跑为 9,942 字符、本变更后为 10,222 字符
（**+280 字符，约 +2.8%**，7 个集合的描述），与 `platform/llm-call-budget` 的既有约束相容。

### 6.4 任务 5.2 未勾选的原因（不谎报）

5.2 要求"意图命中率不低于变更前"，**需要变更前的基线数字，而该基线没有留存**。
实际做到的是：验证了机制完整——`intent_text()` 仍渲染集合描述、动态 KB 节点仍携带
`collection_names`，只是去掉了一段语义重复的文本。**"命中率不下降"这个量化结论无法给出**，
因此不勾选。补齐办法：在归档前先用旧实现跑一遍动态集合命中的评测集并记录命中率，再对比。

