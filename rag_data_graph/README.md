# rag_data_graph —— 跨实体关系语料（供知识图谱通道使用）

补上意图树 `knowledge-entity-relation` 节点所声明的「实体关系」类语料。
此前该通道**没有语料支撑**（`evals/golden/DATA_ISSUES.md` F13）：问它一定拿不到内容，
只能看"工具选没选对"。本目录补的就是这块最小语料。

---

## 一、目录结构

| 路径 | 是否入库 | 说明 |
|------|----------|------|
| `01_销售跨实体关系图谱.md` | ✅ **是** | **唯一真源**：实体清单 / 关联键 / 对账路径 / 已知不一致的判读约定 |
| `lightgraph_ingest/01_销售跨实体关系图谱.txt` | ✅ 是 | 与 `.md` **同内容的纯文本镜像**，供 `light_rag.py::ingest_data_pipeline` 这类"只认 .pdf/.txt"的入口使用 |
| `README.md`（本文件） | ❌ 否 | 说明文档 |

> ⚠️ 改了 `.md` 就必须同步 `.txt`（两者必须逐字一致）。用
> `evals/tools/ingest_graph_corpus.py` 入库时，两个文件都会被识别为同名文档，
> 只入库一次；若要避免手工同步，直接删掉 `.txt`、把 `--src-dir` 指向本目录即可。

---

## 二、这份语料**不做什么**（重要）

- **不重复**任何单表内容，**不新增**任何金额/客户/竞品/日期类事实；
- 只描述「谁和谁通过哪个键关联、怎么对账」，单条事实仍应回到 `raw_data/sales_intel/*.xlsx`
  或 `rag_data/*.md` 去取；
- 对既有语料的已知不一致（F02/F04/F05/F07/F09/F10/F11/F14）**只给检索时的判读约定，不改数据**；
- **不放进 `rag_data/`**：那个目录是 RAG 向量语料库的收录范围（`evals/corpus_rules.py`），
  混进去会改变既有语料清单与 Recall@5 口径。

---

## 三、入库方式（向量库之外的图谱通道）

图谱集合 = LightRAG 的 **workspace**；意图不指定 `collection` 时工具留空，落到默认集合 `default`。

```powershell
# 1) 空跑：看会入库哪些文件、抽多少字（不连图谱、不消耗 LLM）
python -m evals.tools.ingest_graph_corpus --dry-run

# 2) 真正入库到默认集合 default（已入库的同名文件会自动跳过，可安全重跑）
python -m evals.tools.ingest_graph_corpus

# 3) 入库到指定集合（与意图路由硬约束里的 collection 名一致）
python -m evals.tools.ingest_graph_corpus --workspace sales_graph

# 4) 只入库其中一份
python -m evals.tools.ingest_graph_corpus --only-file 01_销售跨实体关系图谱.txt
```

- 入库走 `light_rag.insert_document(workspace, filename, content)`，**与上传接口同一条写入路径**；
- 脚本**不会**调用 `clear_workspace()` 或任何删除能力，不会动既有图谱数据；
- 图谱抽取由 LLM 完成，单篇语料的入库耗时在分钟级；失败可重跑（幂等）。

## 四、与既有语料的一致性

本语料中的表名/sheet 名/字段名/编号规则**逐条对齐**：

- `raw_data/sales_intel/*.xlsx` 各自的「字段字典」sheet；
- `rag_data/市场活动效果.md`（LD ↔ 销售业绩月度表的对账表述）、
  `rag_data/销售分析指标词典与诊断手册.md`（赢单率分母口径）、
  `rag_data/销售方法论与商机资质判定手册.md`（阶段流转 / 45 天搁置 / 90 天公海）。

新增内容为零事实，只有「关联关系」这一层描述。
