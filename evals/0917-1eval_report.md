# Ai_Agent_Frame 评测报告（eval_report）

- 生成时间：2026-09-17 11:52:39
- 阈值版本：`thresholds.version = 1`
- 运行环境：Windows 10 / Python 3.14.6
- 复现命令：`python -m evals.report --run-all`

## 一、结论（先看这里）

❌ **质量门未通过**，共 2 项违规：

- rag.mrr_min 违规：mrr=0.2500 < 0.5500
- cost.avg_tokens_per_turn_max 违规：avg_tokens_per_turn=12299.5000 > 6000.0000

> 说明：阈值未配置的项、以及本次未采集到的指标会被**跳过**而不是判失败，所以「通过」的含义是「已测量的部分达标」。

## 二、四个必测指标

| 指标 | 本次实测 | 阈值 | 阈值版本键 |
| --- | --- | --- | --- |
| 意图准确率 | 1 ✅ | 0.8 | `intent.accuracy_min` |
| 意图准确率（5 条边界样本） | 不可用 | 0.6 | `intent.boundary_accuracy_min` |
| RAG Recall@5 | 1 ✅ | 0.7 | `rag.recall_at_5_min` |
| RAG Hit@5 | 1 ✅ | 0.8 | `rag.hit_at_5_min` |
| RAG MRR | 0.25 ❌ | 0.55 | `rag.mrr_min` |
| 工具调用成功率 | 1 ✅ | 0.75 | `tool.call_success_min` |
| 工具关键参数命中率 | 1 ✅ | 0.6 | `tool.key_arg_recall_min` |
| 单轮 token 成本（端到端） | 12299.5 ❌ | 6000 | `cost.avg_tokens_per_turn_max` |
| P95 延迟 (ms) | 不可用 | 45000 | `cost.p95_latency_ms_max` |
| 答案合格率（生成层） | 不可用 | 0.7 | `answer.pass_rate_min` |
| 应拒答正确率（生成层） | 不可用 | 0.6 | `answer.abstain_correct_rate_min` |

## 三、分组明细

### 3.1 意图分类

- `avg_tokens_per_turn` = 3600.0000
- `boundary_accuracy` = 1.0000
- `boundary_scored` = 1
- `boundary_total` = 1
- `input_tokens` = 0
- `intent_accuracy` = 1.0000
- `intent_scored` = 2
- `llm_calls_per_request` = 1.0000
- `mean_latency_ms` = 18305.9950
- `median_tokens_per_turn` = 3600.0000
- `output_tokens` = 0
- `p95_latency_ms` = 22665.0055
- `total` = 2
- 样本数 = 2
- 真实 LLM 调用 2 次，input=6701 output=499 total=7200 tokens

- ⚠️ **样本量不足**：以下数值仅供参考，**不参与质量门判定**
  - `boundary_accuracy`：本次 1 条，需 ≥5 条。低于下限时该指标在统计上不成立（如 P95 在 n=3 时插值结果几乎等于最大值）
  - `p95_latency_ms`：本次 2 条，需 ≥20 条。低于下限时该指标在统计上不成立（如 P95 在 n=3 时插值结果几乎等于最大值）

### 3.2 RAG 检索

- `hit@5` = 1.0000
- `mean_latency_ms` = 1729.1600
- `mrr` = 0.2500
- `p95_latency_ms` = 2767.5170
- `recall@5` = 1.0000
- `scored` = 1
- `skipped_unanswerable` = 1
- `total` = 2
- 样本数 = 2

- ⚠️ **样本量不足**：以下数值仅供参考，**不参与质量门判定**
  - `p95_latency_ms`：本次 2 条，需 ≥20 条。低于下限时该指标在统计上不成立（如 P95 在 n=3 时插值结果几乎等于最大值）

### 3.3 工具调用

- `avg_tokens_per_turn` = 12299.5000
- `error_rate` = 0.0000
- `key_arg_cases` = 2
- `key_arg_recall` = 1.0000
- `llm_calls_per_request` = 6.5000
- `mean_latency_ms` = 41923.7100
- `median_tokens_per_turn` = 12299.5000
- `memory_cleanup_failures` = 0
- `memory_isolated` = True
- `p95_latency_ms` = 49169.7540
- `tool_success_rate` = 1.0000
- `total` = 2
- 样本数 = 2
- 真实 LLM 调用 13 次，input=21979 output=2620 total=24599 tokens

- ⚠️ **样本量不足**：以下数值仅供参考，**不参与质量门判定**
  - `p95_latency_ms`：本次 2 条，需 ≥20 条。低于下限时该指标在统计上不成立（如 P95 在 n=3 时插值结果几乎等于最大值）

## 四、一个可讲的失败案例

本次没有失败样本（或全部指标未采集）。

## 五、覆盖范围与已知局限

- 黄金集规模：意图 21 条 / RAG 32 条 / 工具 16 条。
- ``--limit`` 为**分层抽样**（见 ``evals/sampling.py``），不是'取前 N 条'：必含边界样本 / 应拒答样本，其余在各层间轮转分配。因此**冒烟跑的数字不能代表全量**，只能验证链路通不通。``--order random`` 会在抽样前先打乱（连带抽样集合一起变化，适合反复冒烟）；``--seed N`` 可锁定同一批用例。
- **样本量不足的指标不判定、只展示**：如 P95 需要 ≥20 条、边界准确率需要 ≥5 条，未达下限时会在分组明细里标注原因并从质量门剔除。这是有意为之——小样本下的分位数测的是偶发抖动，不是整体水位。
- ⚠️ 已知约束：工具黄金集仅 15 条 < 20，**该组 P95 即使全量跑也不进质量门**。如需启用这条红线，请把工具集扩到 ≥20 条。
- `expected_doc_id` 若为 `null`，RAG 召回判定走**文件名通道**兜底（两条通道取较优者），这是设计内的行为。
- RAG 段（`rag`）测的是**检索层**（Recall@5 / MRR）；**生成层**另由 answer 段（`evals/tools/grade_answers.py`）判分，两者互补：召回对了不代表说对了。
- 生成层当前用**规则判分**（要点命中 + 干扰项检测 + 拒答检测），不是 LLM-as-judge。理由是先要有可复现的数字；LLM judge 应作为**对照**叠加。
- 单轮 token 成本来自旁路采集的真实 LLM `usage`；若采集失败该行显示「不可用」，不使用字符数估算。
- 全套评测依赖 Milvus / Redis / 模型 API，**均为本机单机环境**，不代表生产容量。

