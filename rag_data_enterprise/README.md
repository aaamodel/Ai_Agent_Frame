# rag_data_enterprise —— 企业职能类语料补充包

补上 `evals/golden/intent_cases.jsonl` 中**原本没有任何数据支撑**的知识类问题：
`knowledge-hr` / `knowledge-it` / `knowledge-finance` / `knowledge-biz-system` / `knowledge-entity-relation`。

全部内容与现有 `raw_data/sales_intel/*.xlsx` 和 `rag_data/*.md` **逐条对齐**（对齐校验见 `_meta/coverage_map.md` 第二节），
不新增与现有语料冲突的事实。

---

## 一、目录结构

| 路径 | 是否入库 | 说明 |
|------|----------|------|
| `corpus/*.md`（5 篇） | ✅ **是** | 真正参与 RAG 检索的语料，入库时只指向这一层 |
| `lightgraph_ingest/*.txt`（5 篇） | ⚠️ 视通道 | 与 corpus 同内容的纯文本副本，供 **LightRAG 图谱抽取**使用（原因见第三节） |
| `_meta/coverage_map.md` | ❌ 否 | 标注资产：问题 → 文档 → 答案锚点映射 + 一致性校验 |
| `_meta/rag_cases_supplement.jsonl` | ❌ 否 | 可选：8 条 RAG 评测 case（R33~R40），确认后追加进 `evals/golden/rag_cases.jsonl` |
| `README.md`（本文件） | ❌ 否 | 说明文档 |

> ⚠️ 入库时**只能把 `--src-dir` 指向 `corpus/`**。若指向 `rag_data_enterprise/` 根目录，
> 会把 README、映射表一并灌进去（评测问题原句直接进库 → Recall 虚高 + 黄金集污染），
> 且 `corpus/` 与 `lightgraph_ingest/` 的同名内容会被重复收录两次。

---

## 二、五篇语料与覆盖意图

| 文档 | 意图 | 覆盖的问题 |
|------|------|------------|
| `01_人事制度与员工手册.md` | knowledge-hr | I01（离职时限与交接流程）、I02（年假天数） |
| `02_IT支持与网络故障排查手册.md` | knowledge-it | I03（内网中断是否机房故障） |
| `03_财务与发票信息手册.md` | knowledge-finance | I04（开票信息查询路径） |
| `04_内部业务系统使用手册.md` | knowledge-biz-system | I05（采购审批系统移动端能力） |
| `05_组织与系统实体关系图谱.md` | knowledge-entity-relation | I06（CRM↔工单系统）、I07（线索首次筛选部门）、B01（审批流↔财务数据流） |

---

## 三、入库方式

### 1）向量库（Milvus，走 `rag_knowledge_search`）

```bash
# 注意：reingest_corpus.py 的默认 RAG_DATA_DIR 是 app/rag_data（当前仓库不存在该目录），
# 现有销售语料实际在仓库根 rag_data/，所以必须显式传 --src-dir。
python evals/tools/reingest_corpus.py \
  --src-dir rag_data_enterprise/corpus \
  --collection enterprise_kb \
  --chunk-size 512
```

- `--collection` 建议用 `enterprise_kb`（与销售语料的 `sales_kb` 分开，避免稀释销售语义）；
  若希望与销售语料同集合评测，改成 `sales_kb` 即可，此时 `_meta/rag_cases_supplement.jsonl`
  里的 `collection` 字段需同步改成 `sales_kb`。
- 建议先 `--dry-run` 看分块数，再正式写入；**不要加 `--reset`**（会清空整个物理集合）。

### 2）知识图谱（LightRAG，走 `knowledge_graph_search`）

`app/infrastructure/knowledgebase/light_rag.py` 的 `ingest_data_pipeline` **只扫描 `raw_data/` 下的 `.pdf` 与 `.txt`，
不认 `.md`**——所以 md 直接丢进去图谱抽不到实体。做法是把它转成 txt 放进 `raw_data/`：

```bash
# 复制（不要移动，corpus/ 里的 md 仍要供向量库使用）
cp rag_data_enterprise/lightgraph_ingest/*.txt raw_data/

# 重建图谱（会向 LightRAG 重新抽取实体与关系）
python app/infrastructure/knowledgebase/light_rag.py
```

>I06 / I07 / B01 属于 `knowledge-entity-relation`，走的是图谱通道，**只灌向量库不灌图谱的话这三条依然拿不到关系证据**。

---

## 四、关键设计取舍（看这里再决定要不要改）

1. **财务账号不写死**：`03` 文档明确"开户行与账号以 FSS 实时数据为准，本文档不固化具体数字"。
   这是刻意为之——评测考的是"查询路径"，答出具体账号数字反而说明在编造。
2. **内部「工单系统」与在售「工单系统」同名**：`04` 手册第一章设了"命名约定"小节消歧，
   I06 判分时需注意干扰项（5 万 / 12 万是产品定价，不是内部系统关系）。
3. **`05` 要求"关系类问题必须同时答已打通与未打通边界"**，否则 I06/B01 会退化成是非题。
4. **台账「负责人」字段 ≠ 首次筛选部门**：`05` 的 1.3 专门做了口径澄清，这是 I07 的核心判别点。
5. **部门负责人人名**：王强（销售部）、陈静（市场部）取自现有语料与台账，若人事变动需同步改 `05` 的 1.4。

---

## 五、有意不补的条目（补了会破坏用例设计）

| 问题 | 意图 | 原因 |
|------|------|------|
| I08 上证指数、I09 数据跨境新规、B02 七鱼资质 | web-live-info | 判别点正是"内部语料没有、必须联网"，补了用例即失效，且数值必然过期 |
| I10~I12、B03 | data-* | 数据本来就在 `raw_data/sales_intel/*.xlsx`，缺的是表格工具调用，不是语料 |
| I13/I14 | files-* | 目标是仓库真实文件与真实类名（RAGService），已有实体 |
| I15~I17、B04/B05 | task-todo-plan / sys-* | 无知识库诉求 |

---

## 六、与现有语料的一致性（已核对）

- 线索首次筛选 → 市场部（对齐《销售方法论…》1.2 与台账「阶段流转规则」sheet）；
- 阶段顺序、45 天搁置 / 90 天公海、MEDDIC 2/1/0 分制与 A(10~12) 等级 → 全部沿用现有手册口径；
- 编号格式 `LD-YYYYMM-XXX`、`MA-YYYYMM-XX`、区域取值（华北/华东/华南/西南）、
  来源渠道取值（市场活动/官网留资/转介绍/电销/内容营销）→ 全部照抄台账字段字典；
- **不新增**任何竞品、产品定价、客户案例事实，避免与 `rag_data/` 产生新的矛盾（对照 `evals/golden/DATA_ISSUES.md`）。
