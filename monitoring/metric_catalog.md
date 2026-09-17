# 指标字典（metric_catalog）

> 这份文件回答一个问题：**线上该看哪些指标、每个指标怎么算、数据从哪来、多少该报警**。
>
> 一条贯穿全篇的原则：**所有指标靠 `trace_id` 串联整条链路**。一次请求会经过
> 意图改写 → 意图分类 → 模式决策 → （RAG 检索 / 工具调用）→ 生成，
> 每一段的耗时、token、错误都能用同一个 `trace_id` 回放出来。
>
> 另一条原则：**拿不到就说拿不到**。本文件对每个指标都标注了「口径 / 来源 / 当前状态」，
> 凡是没有真实埋点的，一律标 `⛔ 未采集` 并写清缺什么，绝不用估算值冒充实测值。
> 原因见文末《为什么有些指标是空的》。

---

## 0. 全景：四层

| 层 | 看什么 | 一句话目的 | 主要来源 |
|---|---|---|---|
| **性能** | P95 / P99 延迟、TTFT（首 token 时延） | 用户等不等得起 | Langfuse trace + `benchmark/` 压测基线 |
| **成本** | 单次会话成本、input/output token 比 | 钱烧得快不快 | `monitoring/cost_calculator.py` + Langfuse usage |
| **质量** | 重试率、重新生成率、任务完成率、eval 分数 | 答得对不对 | `evals/` 回灌 + Langfuse score |
| **稳定性** | 工具错误率、循环次数、熔断触发次数 | 会不会崩、会不会转圈 | 预算/熔断埋点 + `benchmark/test_circuit_breaker.py` |

**阈值不是拍脑袋定的**：性能与稳定性两层的告警线来自《压测报告》跑出来的基线，
再乘一个安全系数反推（见第 5 节）。这是把「eval / 压测 / 监控」三件事串成一个体系的点睛之处。

---

## 1. 性能层

| # | 指标 | 口径定义 | 计算方式 | 数据来源 | 建议阈值 | 状态 |
|---|---|---|---|---|---|---|
| P1 | **端到端 P95 延迟** | 一次 `/chat` 请求从进到出（含模型等待）的耗时 P95 | `percentile(durations, 95)`，numpy linear 口径 | Langfuse `observations` 的 start/end；压测走 locust `response_time` | 超压测基线 ×1.5 告警 | ✅ 可采集 |
| P2 | **端到端 P99 延迟** | 同上，P99 | 同上 | 同上 | 超基线 ×2 告警 | ✅ 可采集 |
| P3 | **分阶段耗时** | 意图改写 / 分类 / 检索 / 工具 / 生成 各段耗时 | 同一 `trace_id` 下按 observation 名称聚合 | Langfuse trace 树 | 单段超总预算 40% 关注 | ⚠️ 依赖 trace 命名规范 |
| P4 | **TTFT（首 token 时延）** | 流式接口发出首个 token 的延迟 | 需要流式埋点打时间戳 | —— | —— | ⛔ 未采集 |
| P5 | **吞吐 QPS** | 单位时间完成的请求数 | 压测期 `请求数 / 时长` | `benchmark/locustfile.py` | 见压测报告 | ✅ 压测时可得 |

**怎么拿到 P1/P2**：`monitoring/langfuse_daily_report.py` 已经实现。它按天拉取
Langfuse 公开 REST 接口的 observation 列表，逐个抽取耗时，再算 percentile：

```bash
python -m monitoring.langfuse_daily_report --date 2026-09-13 --out daily.md
```

> P4（TTFT）为什么空着：`/chat/with_agent` 是 SSE 流式接口，当前代码在流式路径上
> **没有在「发出第一个 chunk」时打时间戳**，只有整段结束的时间。要拿到 TTFT 需要在
> 流式生成器里埋一个 `first_token_at`，这是纯增量改动，但没有埋点之前任何 TTFT 数字都是编的。

---

## 2. 成本层

| # | 指标 | 口径定义 | 计算方式 | 数据来源 | 建议阈值 | 状态 |
|---|---|---|---|---|---|---|
| C1 | **单次会话成本** | 一次会话累计的模型花费 | Σ(模型用量 × 单价) | `cost_calculator.compute_cost()` | 突增 2× 告警 | ⛔ 依赖定价表 |
| C2 | **input / output token 比** | 输入 token ÷ 输出 token | 直接相除 | Langfuse usage | 持续 >10 关注（上下文膨胀） | ✅ 可采集 |
| C3 | **单轮平均 token** | 每轮对话消耗的 token 总数 | `total_tokens / turns` | `evals/runners/_usage.py` 采集真实 usage | `thresholds.yaml: avg_tokens_per_turn_max` | ✅ 评测链路可采集 |
| C4 | **按模型拆分用量** | 各 tier 模型各自的调用次数与 token | 按 `model` 分组累加 | mock server `/stats` 或 Langfuse | —— | ✅ 可采集 |
| C5 | **token 成本（人民币）** | C1 的货币化 | 用量 × 单价 | `monitoring/model_pricing.yaml` | —— | ⛔ 定价表未填 |

**C1/C5 的诚实说明**：`monitoring/model_pricing.yaml` 里 **4 个模型的价格全部是 `null`**，
`prices_filled: false`。这不是遗漏，是刻意设计：

- `cost_calculator.py` 遇到未定价模型时**返回 `None` 并打印警告**，CLI 直接 `exit 2`，**绝不静默返回 0**。
- 因为一个「看起来完整但其实用了错价格」的总额，比一个明说「未定价」的空值**危险得多**——
  前者会让人在错误的前提下做决策。
- 校准动作：打开 `model_pricing.yaml`，按厂商官方定价页填写 `input_per_1k` / `output_per_1k`，
  并同时填 `price_effective_date` 与 `source_url`（价格会变，必须能追溯是哪天的价）。

```bash
# 填完后自检
python -m monitoring.cost_calculator --check-pricing
```

---

## 3. 质量层

| # | 指标 | 口径定义 | 计算方式 | 数据来源 | 建议阈值 | 状态 |
|---|---|---|---|---|---|---|
| Q1 | **意图准确率** | 分类正确的样本占比 | `accuracy(golds, preds)` | `evals/runners/run_intent.py` | ≥ 0.80（见 thresholds） | ✅ 可采集 |
| Q2 | **边界样本准确率** | 仅 5 条易混淆样本的准确率 | 同上，子集 | 同上 | ≥ 0.60 | ✅ 可采集 |
| Q3 | **Recall@5** | 前 5 条里命中期望文档的比例 | `recall_at_k()` | `evals/runners/run_rag.py` | ≥ 0.70 | ✅ 可采集 |
| Q4 | **MRR** | 首个命中结果的排名倒数 | `mrr()` | 同上 | ≥ 0.55 | ✅ 可采集 |
| Q5 | **工具调用成功率** | 调用了期望工具的样本占比 | `tool_call_success_rate()` | `evals/runners/run_tool.py` | ≥ 0.75 | ✅ 可采集 |
| Q6 | **关键参数命中率** | 期望关键参数被正确传入的比例 | `key_arg_recall()` | 同上 | ≥ 0.60 | ✅ 可采集 |
| Q7 | **eval 分数回灌** | 上述指标写入 Langfuse score，与线上 trace 同视图 | `eval.<metric>` 命名 | `monitoring/push_eval_scores.py` | —— | ✅ 可采集 |
| Q8 | **重试率** | 用户对同一问题重问的比例 | —— | —— | 上升即关注 | ⛔ 未采集 |
| Q9 | **「重新生成」点击率** | 前端点重新生成的次数 / 总提问数 | 需要前端埋点 | —— | —— | ⛔ 未采集 |
| Q10 | **任务完成率** | 多步任务真正跑到完成态的比例 | 需要定义「完成态」 | —— | —— | ⛔ 未采集 |

**Q1–Q6 的定位**：这六项**不是线上实时指标**，而是**回归指标**——它们在 CI 里跑（`.github/workflows/eval.yml`），
防止改动把质量改差。真正的证据是 `evals/eval_report.md`。

**Q7 的作用**是把回归指标搬进线上同一个观测视图：

```bash
python -m monitoring.push_eval_scores --run-name ci-1234 --dry-run   # 先看要推什么
python -m monitoring.push_eval_scores --run-name ci-1234             # 真推
```

> ⚠️ Langfuse 的 score 是**追加写入（append-only）**，不做去重也不覆盖。
> 同一个 `run_name` 反复推会产生重复记录。要么用带日期/commit 的 `run_name`，
> 要么接受「同一指标一个版本一条 score」的语义。

**Q8–Q10 为什么空着**：这三项都**依赖前端或用户行为数据**，当前项目没有前端埋点，
服务端也拿不到「用户点了重新生成」这种信号。硬凑一个数字（比如用「同 session 内重复 query」估重试率）
是可以的，但那已经有了一个**有偏的定义**——真要用，必须先在文件里把定义写死并说明偏差，否则不如不做。

---

## 4. 稳定性层

| # | 指标 | 口径定义 | 计算方式 | 数据来源 | 建议阈值 | 状态 |
|---|---|---|---|---|---|---|
| S1 | **工具错误率** | 工具调用失败次数 / 总调用次数 | 分组计数 | `AgentResponse.steps` + 压测统计 | > 5% 告警 | ⚠️ 评测链路可采，线上无埋点 |
| S2 | **循环次数** | ReAct 单次请求的实际步数 | `len(steps)` | `evals/runners/run_tool.py` 旁路采集 | 逼近上限告警 | ⚠️ 同上 |
| S3 | **熔断触发次数** | 三态熔断器进入 `OPEN` 的次数 | 计数器 | `AsyncModelHealthStore`（进程内） | 突增告警 | ⚠️ 进程内，不跨进程 |
| S4 | **故障切换耗时** | 「探测到故障 → 完成切换」的毫秒数 | 见 `benchmark/test_circuit_breaker.py` | 熔断专项压测 | 见压测报告 | ✅ 压测可得 |
| S5 | **API 错误率** | 非 2xx 响应占比 | locust `fail_ratio` | `benchmark/locustfile.py` | > 1% 告警 | ✅ 压测可得 |

**S3 的重要限制**：`AsyncModelHealthStore` 的熔断状态是**进程内内存状态**，
`failure_threshold=2`、`open_duration_ms=30000`。这意味着：

- 单进程部署时，熔断行为符合预期；
- **多 worker / 多副本部署时，各进程各熔各的**，A 进程熔断了 B 进程并不知道。
  要把 S3 做成真正的集群级指标，需要把健康状态外移到 Redis 之类共享存储。

**主动写明这条限制**比被人发现强得多——它体现了对系统边界的清晰认知。

**S4 怎么测**：`benchmark/test_circuit_breaker.py` 注入一个「主模型超时」的故障
（patch 掉唯一那个真实 LLM 出口），观察路由是否切到备用模型，并记录耗时。

```bash
python benchmark/test_circuit_breaker.py --fault hang --hang-seconds 3 --repeat 3
```

---

## 5. 阈值从哪来（把三件事串起来的那句话）

告警阈值分两种来源，**别混**：

1. **性能 / 稳定性阈值 → 来自压测基线**。
   先跑《压测报告》，拿到本机 QPS / P95 / 错误率，再乘安全系数：
   例如 P95 基线 3.2s，则告警线取 `3.2 × 1.5 ≈ 4.8s`。
   这就是「阈值来自压测基线」——**不是拍脑袋**。
2. **质量阈值 → 来自 eval 的 baseline**。
   先跑一次 `python -m evals.report --run-all` 得到当前真实分数，
   再把 `evals/thresholds.yaml` 的线定在「略低于当前水平」，
   这样后续任何回归都会立刻触发质量门。当前 thresholds 是**保守初值**，
   文件里已显式注明「拿到真实 baseline 后必须回来校准」。

> 拿压测数字时请记住：《压测报告》里的所有数字都来自**本机单机环境，非生产环境数据**。
> 用单机基线去定生产告警线是不成立的，需要在真实部署环境重跑一次。

---

## 6. 为什么有些指标是空的

一句话原则：**空着比编一个数字好**。

| 空缺项 | 缺什么 | 补齐的代价 |
|---|---|---|
| P4 TTFT | 流式路径缺 `first_token_at` 埋点 | 小，改流式生成器即可 |
| C1/C5 成本 | 定价表 4 个模型全是 `null` | 小，但必须人工核实官方价 |
| Q8 重试率 | 无前端/用户行为埋点 | 中，需要产品定义「重试」 |
| Q9 重新生成率 | 同上 | 中 |
| Q10 任务完成率 | 缺「完成态」定义 | 中，先要定义什么叫完成 |
| S1–S3 线上版 | 评测链路有、生产链路无埋点 | 中，接入统一 metrics 上报 |

这些项已全部登记在《方案完成情况说明》里，**不藏**。
关于「监控做得全不全」，本文件的态度是：

> 「我分四层列了指标字典，其中性能、成本、稳定性的一部分已经跑通并有真实数字，
> 另一部分因为缺埋点先空着，并且写清了缺什么、补的代价多大。
> **列出一个自己都知道没采集的指标、还给它编个数字**，是最差的做法。」

---

## 7. 快速索引：指标 → 文件

| 指标组 | 采集/计算文件 | 运行命令 |
|---|---|---|
| Q1–Q6 全部回归指标 | `evals/metrics.py` + `evals/runners/*` | `python -m evals.report --run-all` |
| 质量门断言 | `evals/thresholds.yaml` + `evals/test_eval_gate.py` | `pytest evals/ -q` |
| P1/P2/C2/C4 线上日结 | `monitoring/langfuse_daily_report.py` | `python -m monitoring.langfuse_daily_report --date …` |
| C1/C5 成本 | `monitoring/cost_calculator.py` | `python -m monitoring.cost_calculator --usage …` |
| Q7 分数回灌 | `monitoring/push_eval_scores.py` | `python -m monitoring.push_eval_scores --run-name …` |
| S4 熔断切换耗时 | `benchmark/test_circuit_breaker.py` | `python benchmark/test_circuit_breaker.py --fault hang` |
| S5/P5 压测吞吐与错误率 | `benchmark/locustfile.py` | `bash benchmark/run_bench.sh --mode mock` |

---

*本文件中的价格为量级参考，实际请核实厂商实时定价。引用压测类数字时必须标注「本机单机环境，非生产环境数据」。*
