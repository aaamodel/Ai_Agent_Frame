# 评测 / 压测 / 监控 使用文档（总索引）

三份文档按板块拆开，本页是索引 + 一页速查。

| 板块 | 文档 | 解决什么问题 |
| --- | --- | --- |
| **评测** | [`01_评测使用文档.md`](./01_评测使用文档.md) | 效果好不好、改了有没有变差 —— **重点，最详细** |
| **压测** | [`02_压测使用文档.md`](./02_压测使用文档.md) | 抗多少并发、多快、多贵、挂了多久切走 |
| **监控** | [`03_监控使用文档.md`](./03_监控使用文档.md) | 上线后每天花多少钱、质量有没有掉 |

三者的关系是一条链：**评测定基线 → 压测定容量 → 监控守线上**。
监控里的 `push_eval_scores.py` 会把评测分数推到 Langfuse，让「每次上线后的质量」和「日常成本/延迟」出现在同一张图里。

---

## 一、30 秒速查：我要的数字从哪来

| 你要的数字 | 跑什么 | 结果在哪 |
| --- | --- | --- |
| 意图准确率 / 边界样本准确率 | `python -m evals.report --run-all` | `evals/eval_report.md` |
| Recall@5 / Hit@5 / MRR | 同上 | 同上 |
| 工具调用成功率 / 关键参数命中率 | 同上 | 同上 |
| 单轮 token / P95 延迟 | 同上（端到端口径取工具组） | 同上 |
| 答案合格率 / 应拒答正确率 | `python evals/tools/grade_answers.py --answers <预测答案.jsonl>` | `evals/_results/answer.json` |
| chunk 512 vs 1024 谁更好 | `python -m evals.run_chunk_regression --ingest` | `evals/chunk_regression_report.md` |
| QPS / P95 / 错误率 | `bash benchmark/run_bench.sh 50 10m --mode real` | `benchmark/_results/locust_real_50_stats.csv` |
| 熔断切换耗时 | `python benchmark/test_circuit_breaker.py --repeat 2 --fault hang` | `benchmark/熔断切换实测.md` |
| 昨天花了多少钱 | `python monitoring/langfuse_daily_report.py --date 2026-09-13` | stdout / `--out` |

---

## 二、前置依赖矩阵（哪些命令现在就能跑）

这是最容易踩的坑：**大部分脚本不需要任何中间件，但少数必须 Milvus / Redis / 模型 API 在线**。
对照这张表，先跑绿灯区把流程走通，再起环境。

| 依赖 | 评测 | 压测 | 监控 |
| --- | --- | --- | --- |
| 无（纯离线） | 单测、黄金集体检、rubric 自检、语料清单、分块计划、答案判分 | mock LLM 服务、mock 压测、熔断 `--dry-run` | 价格表检查 `--check-pricing` |
| 需要 Milvus | 语料入库、RAG 评测、chunk 回归 | 真实压测（若走知识库） | — |
| 需要 Redis | 意图/工具评测（会话与熔断状态） | 真实压测 | — |
| 需要模型 API | 意图 / 工具评测、答案生成 | 真实压测 | — |
| 需要 Langfuse | — | — | 日报、评分推送 |

> **当前环境状态**：Milvus(19530) / Redis(6379) / PG(5432) / app(8000) 均未启动，`locust` 未安装。
> 所以上面所有「真实」类数字目前都是 `【待填】`，**这是刻意的，不是漏了**——不编数字是本套方案的第一原则。

---

## 三、推荐执行顺序

```
第 0 步  跑离线绿灯区，确认工具链没坏        （不需要任何中间件）
第 1 步  起 Milvus + Redis + app
第 2 步  入库评测语料 → 跑 baseline → 校准 thresholds.yaml
第 3 步  chunk 512 vs 1024 回归实验
第 4 步  mock 压测（零成本）→ 真实压测（梯度爬坡）
第 5 步  熔断专项
第 6 步  填 model_pricing.yaml → 跑 Langfuse 日报
第 7 步  回填简历里的 [X]
```

✅ **第 2 步的前置数据缺陷已修复（2026-09-14）**：原意图黄金集 15 条里有 10 条的 query 与
`intent_tree.py` 的 `examples` 逐字相同（`examples` 在推理时既进向量索引又进 LLM prompt，
等于拿训练集当测试集）。现已**删除全部示例派生条目并重写 held-out**（意图 22 条、
RAG 32 条、工具 15 条），并同批排查修复了 9 条同类污染（P1–P9）——其中一条是新发现的：
`app/rag_data/other/销售助手评测问题集.md` 原本被递归灌进检索库，会让 Recall@5 系统性虚高。

改完黄金集**必跑**：

```bash
python -m evals.tools.check_golden_purity      # 有 FAIL 返回 1
```

明细见 [`evals/golden/DATA_ISSUES.md`](../evals/golden/DATA_ISSUES.md)，
以及 [`01_评测使用文档.md`](./01_评测使用文档.md) 第 7 节。

---

## 四、贯穿三套工具的四条硬规则

1. **拿不到就写「不可用」，不用估算值填**。未定价的成本返回 `None` 而非 `0`；未采集的指标跳过而非判失败。
2. **阈值是配置不是代码**。全在 `evals/thresholds.yaml` / `monitoring/model_pricing.yaml`，改数字不改代码。
3. **删表必须显式确认**。评测链路 `overwrite` 恒为 `False`；唯一允许删表重建的入口是 `reingest_corpus.py --reset --confirm-reset`。
4. **所有压测数字必须标注「本机单机环境，非生产环境数据」**。

---

## 五、文件总数

| 板块 | 文件数 | 目录 |
| --- | --- | --- |
| 评测 | 21 | `evals/` |
| 压测 | 8 | `benchmark/` |
| 监控 | 6 | `monitoring/` |
| CI | 1 | `.github/workflows/eval.yml` |

`evals/` 里有 4 个文件不属于原始清单（`doc_ids.py`、`reingest_corpus.py`、`run_chunk_regression.py`、`answer_quality.py` 及其工具），缺了它们任务闭环不了。

---

## 六、设计提案与变更规范

| 文档 | 内容 |
| --- | --- |
| [`05_技术提案_计划执行控制与目标锚点.md`](./05_技术提案_计划执行控制与目标锚点.md) | 计划台账、跳过/提前收尾控制协议、`agent_goal` 目标锚点的设计取舍、风险与实现落点 |
| [`04_优化纪要_面试版.md`](./04_优化纪要_面试版.md) | 优化过程与结论纪要 |
| [`../openspec/`](../openspec/) | 规范驱动变更：`proposal.md` / `design.md` / `tasks.md` / `specs/**/spec.md` |

> 评测 / 压测 / 监控三套工具本身也在 `evals/`、`benchmark/`、`monitoring/` 下受单测覆盖，随 `pytest 测试 evals benchmark -q` 一并回归。
