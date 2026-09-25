# Design: Evidence Board（证据板）

## Context

动机与范围见 [proposal.md](./proposal.md)，行为契约见 [specs/agent/evidence-board/spec.md](./specs/agent/evidence-board/spec.md)。此处只列影响方案的现状事实：

- ReAct 执行在 [execute_node.py](../../../app/core/agent/graph/nodes/execute_node.py) 内分两套循环：函数调用协议（FC，主链路）与文本协议（降级备胎）。plan_execute 模式的子任务提炼是第三条 LLM 调用路径。
- 已存在 v0 压缩 `compact_tool_observations`（[_common.py](../../../app/core/agent/graph/nodes/_common.py)）：最近 3 条 tool 消息全文、更早的压成 200 字首段短桩——纯位置型，不看内容。它是本次的替换对象，同时保留为降级 fallback。
- 原始观测已全文存在请求状态里：ReAct 的 `react_messages`（经 Redis checkpointer 持久化）、plan 的 `subtask_results[*].observation`（截 8000 字）。证据外部化的"存储"已经存在，无需新建。
- 工具统一返回纯字符串，但格式按工具约定分化：RAG 为 `--- 知识库检索结果 ---` 头 + `[n] 来源文献/内容片段` 块；文件类为行/`file:line:` 块；SQL/报表为 markdown 表格；错误为 `Error:` 前缀短串。
- 工具执行签名只有 `**kwargs`，拿不到请求上下文；但节点层已有在工具分发前拦截的先例 `gate_tool_approval`——回取能力照此在节点层实现，不进工具注册表。
- jieba 已在依赖中（step_correction 使用）；不得新增第三方依赖。
- FC 发送消息由 `_fc_assemble_messages` 组装，OpenAI 协议要求 tool 消息与 tool_call 配对且 content 非空。

## Goals / Non-Goals

**Goals:**

- 3 轮 RAG（每轮 5 长片段）夹具下，后续轮次发送视图 input token ≤ 改造前基线的 40%，同时夹具标注的关键证据 100% 在板。
- 证据筛选全程确定性、零额外 LLM 调用、零外部工具重调、零新存储/新依赖。
- 一套管道同时服务 FC、文本协议、plan 提炼三条发送视图。
- 证据层任何故障自动退回现状压缩，主链路行为不劣化于今天。

**Non-Goals:**

- 不做跨请求/跨会话的证据复用与长期证据库。
- 不引入向量模型/embedding 做语义相似度（SimHash + n-gram 足够，且保持零依赖与可离线单测）。
- 不改变 ReAct 控制流：不新增"显式综合轮"，不依据"无新增证据"自动终止（单循环，模型自行决定何时作答）。
- 不做 LLM 摘要式压缩（讨论否决：会在压缩点引入幻觉与一次额外调用）。
- 不改 plan_execute 的历史结论台账（模型自产结论本就紧凑），只改本步观测的 6000 字盲截断。
- 不做 plan 提炼的**当轮即时**回取（该次调用无工具通道，不重构成多轮 agentic 循环）；改为"延迟一步"：模型在提炼输出里声明想看的证据编号，节点插入零外部调用的内部恢复步回填后再提炼（见 D10）。

## Decisions

### D1. 总体架构：发送视图管道 + state 内嵌证据层

新增包 `app/core/agent/evidence/`：

| 模块 | 职责 | 性质 |
|---|---|---|
| `models.py` | `EvidenceUnit` 数据模型、kind 枚举、轮次报告结构 | 纯数据 |
| `chunkers.py` | 按工具名分派的切分器注册表 + 通用兜底切分器 | 纯函数 |
| `pipeline.py` | 查询表示、打分、SimHash/Jaccard 去重、预算选择、入管编排 | 纯函数 |
| `view.py` | 单元属主渲染、证据板/索引/尾注渲染（文本与 FC 两版） | 纯函数 |
| `fetch.py` | 编号解析、回取坐标定位、相邻窗口取块 | 纯函数（读 state） |

数据流：

```
工具 observation 字符串
  → ingest（chunkers 切分 → pipeline 打分/去重/选择）
  → state.evidence_units（追加）+ evidence_meta（替换）
  → 每轮组装消息时 view 从 units 重算证据板视图
```

**为什么不选"独立 EvidenceService + Redis/SQLite 外部存储"**：问题边界是请求内膨胀，外部存储引入生命周期/清理/一致性负担而无需求支撑（YAGNI）。**为什么不选"只给 RAG 加句选"**：建不起跨工具统一模型，plan 路径接不上，与完整设计目标冲突。

### D2. Unit 模型与切分

- 字段：`uid`（e1、e2… 请求内递增）、`tool_name`、`round_idx`、`block_idx`、`text`（块原文 ≤600 字）、`source`、`ref`（回取坐标）、`kind`（content/error/status/table）、`score`、`simhash`、`selected`、`also_from`、`fetched`，另加 `exempt`（bool，短观测豁免标记，见 D5a）。
- 切分器注册表 `CHUNKERS: dict[str, callable]`，未注册工具走 `default_chunker`（双换行分段→句子滑窗，重叠 1 句）。
- RAG：解析头部与 `[n]` 块，来源文献作 source；SQL/报表：整张 markdown 表格为 1 个 table 单元（表头行不可分割，行超 `TABLE_MAX_ROWS=30` 截断并保留"共 N 行"）；file/grep/搜索类按各自条目边界切。
- `Error:`/空结果 → error/status 单元；chunker 内部 ≤ `SHORT_OBS_CHARS=200` 不切。注意豁免判定在 ingest 层、阈值更宽（`EXEMPT_OBS_CHARS=300`，见 D5a）：200–300 字观测即使被 chunker 分段，ingest 也会在豁免分支整体建一个单元。
- 常量：`MAX_UNIT_CHARS=600`，中英文句边界正则。

**取舍**：表格不拆行——拆开后表头语义丢失、选择器可能选出无表头的孤立行；整表选择或只留"N 列 M 行+列名"索引。

### D3. 打分（固定权重，0–100）

- 查询表示：jieba 分词（保留 ≥2 字词与单字英文/数字 token）+ ~80 词内置停用词表 + bigram。打分词集 `Q = 本轮 query 词（权重 1.0）∪ 用户原始问题词（权重 0.6）`。
- 信号：BM25（k1=1.5, b=0.75，语料=本请求全部单元）归一 40 分；bigram 连续子串命中封顶 20；数字/日期/百分比/英文代号实体交集封顶 15；结构先验（表格/标题 +、纯导航目录 −）封顶 10；新鲜度 `15 × (1 − round/max_round)` 封顶 15。
- 上板阈值 `S_MIN=35`。error/status 不打分。
- 权重/阈值集中为常量；测试断言排序关系而非绝对分值。

**为什么保留 15 分新鲜度**：纯位置截断已被否决，但"内容相当时新轮次优先"只作平分 tiebreak，旧证据分高仍在板（MMR 与相关性主导）。

### D4. 去重

- 先 sha1 精确短路；再 64 位 SimHash（jieba 词项 + 2-gram 特征，纯 Python）海明距离 ≤3；中文短块（<50 字）SimHash 不稳定，附加字符 3-gram Jaccard ≥0.85 且长度差 <20% 双判。
- 判重保留高分者、同份保留新者；败方 source/ref 并入胜方 `also_from`。
- 每轮只比对"本轮新单元 × 板上存量"，规模几十块内 O(新×存)，不做 LSH。

### D5. 按轮收紧的预算选择

每轮从全量单元重选（非追加）：

- 字符预算 `min(1200 + 400×(round−1), 2400)`；条数 `min(4 + round, 10)`（首轮 5 条，第 4 轮起封顶 10）。
- 选择：过阈值候选排序 → MMR（λ=0.7，相似度用字符 3-gram Jaccard）打散 → 条数内按字符预算装填；超预算单元在句边界截断并标"已截断"。
- 每轮强制保活：最近一轮全部 error/status 行；低分单元一行索引；尾注（总量/上板/低相关/重复计数 + 回取提示）。
- `no_new_evidence`：本轮新单元全判重或全低分（且无 error）。只用于 trace 与尾注提示，不参与控制流。

### D5a. 短观测豁免通道（v2 修订）

动机：file_list / grep / 短 SQL 表 / 错误串等短观测是后续选工具的"导航地图"，词面打分对文件名/路径类内容反而不稳（同义词文件名会被错杀）；它们体量小，没有筛选必要，应无条件可见。

- 常量：`EXEMPT_OBS_CHARS = 300`（按 strip 后字符数，与全板字符口径一致，不引 tokenizer）、`EXEMPT_KEEP_ROUNDS = 3`、`EXEMPT_STUB_HEAD_CHARS = 60`（桩内首句预览长度）。
- 入管：整条观测为非 error/status 且 `len(text.strip()) ≤ 300` → 跳过 chunker，直接建 **1 个 content/table 单元**并标 `exempt=True`、`score=0`。豁免单元 MUST NOT 进入打分、SimHash 判重与 MMR 选择，也不占 D5 的字符/条数预算。
- 精确去重：豁免单元仍计算归一化文本的 sha1，与"已确立单元"（含其他豁免单元）做**精确短路判重**；命中则置 `dupe_of`，按普通判重单元渲染一行重复指针，**不占用豁免原文名额**（防止反复列同一目录刷满 3 个保留位）。不做 SimHash/Jaccard 模糊判重——短文本指纹不稳，模糊判重只用于长块。
- ReAct 视图（FC 与文本协议同构）：属主观测的豁免单元在 `current_round - round_idx < 3`（即产生轮 r 及之后 r+1、r+2 共 3 个轮次）MUST 原文直出；从 r+3 轮起渲染一行桩：
  `[uid] 工具名 · 原始 N 字符 · 首句预览…（需要全文可调取证据编号）`。桩单元与低分索引同区呈现，尾注计数单列"豁免桩"数。3 轮豁免原文总量上界 = 3 × 300 = 900 字符，有界。
- plan 视图：豁免只在"本步"有意义（跨步由结论台账承载）：`render_plan_observation` 的原文直出阈值从 `SHORT_OBS_CHARS=200` 统一改为 `EXEMPT_OBS_CHARS=300`；本步豁免单元原文呈现，不存在"3 轮后桩化"。
- 回取：桩里的 uid 是普通证据编号，ReAct 走 D7 的 `fetch_evidence` 取回，无特殊通道；豁免单元的 ref 指向其属主 tool 消息/子任务记录，取回逻辑不变。

**取舍**：阈值按字符不按 token——管道全程字符口径、可确定性单测；中文 1 字≈1 token，300 字约等于用户期望的 300 token 量级。cl100k_base 对中文与 GLM 词表都不准，不引入。

### D6. 三条发送视图

- **FC**：不新建消息（保 tool_call 配对）。`_fc_assemble_messages` 逐条 tool 消息按其 tool_call_id 找属主单元，按 D5a 豁免规则优先渲染（3 轮内豁免单元原文、超期豁免桩），其余：在板→编号+来源+原文（截断标注）；判重→一行"与 [e?] 重复（另见…）"；低分→一行索引；error/status→一行。聚合尾注只追加到最后一条 tool 消息末端。system 与首轮 user（含技能全文）原样不动 → 前缀缓存友好。
- **文本协议**：`compact_history_lines` 的"工具/结果/动作"块中 `结果:` 段改用同一属主渲染（含豁免规则），尾注挂最后一个工具块。回取指令从模型动作行/JSON 解析，复用同一拦截器。
- **plan 提炼**：[execute_node.py L584](../../../app/core/agent/graph/nodes/execute_node.py) 的 `obs_str[:6000]` 改为：≤300 字原文直出（D5a）；超长则渲染当前子任务（round_idx=子任务序）入选单元 + 落选"已省略"索引 + 总数行；打分 query = 子任务描述 + agent_goal + 用户问题。省略索引措辞为"已省略（低相关；如需全文，在 `requested_evidence_uids` 填入对应编号）"——延迟一步回取见 D10，不在本视图内即时取回。
- 入管时机：三处拿到观测字符串后立即入管（error/empty 也入），随节点 update 一次性写 state，不增加图轮次。

### D7. fetch_evidence

- 定义：两 ReAct 循环组装消息前向 tools 列表去重追加一条 function 定义（uid 必填、window 可选 0–5），不写进 planner 阶段的 `fc_tool_definitions`，不注册进工具注册表。
- 拦截：工具分发前判 `tool_name == "fetch_evidence"`：window=0 且未截断直接返回 Unit.text；否则用 `ref`（tool_call_id/round/block）定位原 tool 消息全文重跑 chunker 取目标块与相邻块——纯本地、零网络。
- 取回结果作为普通 tool 消息进历史；单元标 `fetched=True`，下轮选择 +10 优先且至少保留一轮，同 uid 重复回取不重复占预算。
- 非法 uid/越界 window → 确定性错误文本（含可用编号），与工具错误同形。

### D8. state 与持久化

```python
evidence_units: List[Dict[str, Any]]  # operator.add 追加
evidence_meta: Dict[str, Any]         # 整体替换：{next_seq, rounds:[...]}
```

证据板本身不落 state（每轮重算的视图，避免副本漂移，与 plan 台账"由计划与记录推导"的既有原则一致）。Unit.text 只存切块；完整观测仍在 `react_messages` / `subtask_results`，回取坐标指向它们。

### D9. 降级与可观测

- `ingest`/view/fetch 全包包 try 边界：异常 → warning 日志 + `evidence.fallback` trace + 该轮退回 `compact_tool_observations` 现逻辑（FC）/等宽文本短桩（文本协议）/`[:6000]`（plan）。
- trace：`evidence.round`（new/duplicated/low/on_board/board_chars/no_new_evidence/exempt）、`evidence.fetch`（uid/window/chars/hit）、`evidence.restore`（plan 延迟恢复：task_id/uids/chars/accepted/rejected）、`evidence.fallback`（stage/error）。

### D10. plan_execute 延迟一步回取（v2 修订）

distill 是无 tools 的单发 chat，无法像 ReAct 那样即时回取；采用"模型在提炼输出里声明 → 节点插入内部恢复步 → 再提炼"的延迟通道。

- **协议字段（与候选资产选择严格分离）**：`SubTaskOutcomeSchema` 新增可选字段
  `requested_evidence_uids: Optional[List[str]] = None`（元素为证据编号字符串，`max_length=3`）。
  它与既有 `selected_alternative_id` 语义无关：后者只用于从「可选的替代方向」候选块里选资产/工具方向（`asset:`/`tool:` 标识）；前者只用于索取**本步工具观测中被省略证据单元的全文**。distill 提示词用一条简短规则告知区别，不展开解释：
  > `selected_alternative_id`：需要换数据源/换工具时，从候选列表选一个方向；`requested_evidence_uids`：本步某条被省略的工具结果需要看全文时，填其证据编号（可多条）。两者互不影响，都不需要时留空。
- **合法性校验（节点层，确定性）**：仅接受 `round_idx == 当前 cursor`、kind ∈ content/table、且出现在本步"已省略"集合中的 uid；越步编号、错误/在板单元、伪造编号一律拒绝并计入 `evidence.restore.rejected`，MUST NOT 中断 distill 结论的正常应用。
- **恢复步形态**：受理后在 `cursor + 1` 插入一个内部计划项：id `correction_evidence_{subtask_id}`（撞名追加 `_2`），`action_type="tool"`，工具名为内部伪工具 `evidence_restore`（**不注册进工具注册表**），`action_input={"uids": [...]}`，描述写明"回填本步被省略证据原文（内部步，零外部调用）"。
- **拦截与回填**：plan 工具分发处在调用真实工具前判 `tool_name == "evidence_restore"`（照搬 ReAct 侧 `fetch_evidence` 的节点拦截先例）：按 uid 从 `evidence_units` 取 Unit.text（truncated 单元经其 ref 从 `subtask_results` 已保存观测重切补全，受既有 8000 字存储上限约束），合成 observation（每块带来源抬头与编号）作为恢复步的 `obs_str`；MUST NOT 经过工具注册表、MUST NOT 产生任何外部调用。回填内容不再二次入管（避免自我打分/判重）。
- **再提炼**：恢复步随后走正常 distill——模型在同结构提示词里看到回填原文并产出/修订结论；恢复步的提示词 MUST NOT 再出现"可省略索引索取"措辞，且其输出中的 `requested_evidence_uids` MUST 被忽略（每子任务最多恢复一次，防循环）。
- **护栏复用**：恢复步计入 `step_corrections` 配额与留痕；其 id 以 `correction_` 开头，自动受"纠偏任务不再触发二次纠偏"链式闸门约束；恢复步本身失败按步级错误记账，不影响后续计划项。
- **零字段膨胀方案的否决记录**：曾考虑复用 `selected_alternative_id` 加 `evidence:e7` 标识——否决，两类决策耦合会让模型在"换数据源"和"看本步全文"之间互斥误选，且需把证据项塞进候选块、放宽 evidence_gap 注入闸门；独立字段语义干净、注入无条件（省略索引本就渲染）。

## Risks / Trade-offs

- **确定性打分对同义不同词的召回弱于向量检索** → 原始问题词全轮并入 + bigram + 实体正则三重缓解；尾注的低分索引与 fetch_evidence 给出补救通道；夹具验收同时卡压缩率与关键证据召回。
- **SimHash 对短中文误判** → 短块双判（海明 + Jaccard + 长度差），并有"不重复内容不误杀"单测。
- **证据板每轮全量重算的 CPU 开销** → 单元规模 ≤ 几十块、BM25 语料即本请求单元、无 embedding，实测在毫秒级；若未来单请求观测块数量级上升，再引入增量索引（Open Question）。
- **RAG 文本块缺稳定 chunk_id** → ref 以 tool_call_id + block_idx 回退定位，回取文本由原始 tool 消息重切保证一致性。
- **模型学会滥用 fetch_evidence 拖慢轮次** → window 封顶 5、回取零外部成本（不花钱只花 token），且取回内容同样受下一轮预算约束；trace 可监测回取频次。
- **chunker 与工具输出格式耦合** → 注册表隔离，格式漂移时单个 chunker 坏只影响该工具的归一化（降级为通用兜底切分器），不炸管道。
- **短观测豁免削弱压缩率** → 豁免只保 3 轮且总量有界（≤900 字符），超期即桩化；精确去重防重复目录占名额。3 轮 RAG 长片段夹具不受影响（超 300 字走正常证据板），40% 验收口径不变。
- **plan 延迟恢复多一次 distill 调用** → 条件触发而非每步必发；每次 ≤3 个单元、回填有界；每子任务最多一次且复用纠偏配额；`evidence.restore` trace 监测滥用频次。

## Migration Plan

1. 特性开关 `enable_evidence_board`（默认关），配置关闭时三处视图走现状逻辑，零行为变化；T1–T3 合入后在测试环境打开验证。
2. 夹具验收（token ≤ 基线 40%、关键证据全在板、383 全量测试不回归）通过后默认打开；保留开关一个迭代用于快速回滚。
3. 回滚：关开关即恢复现状；state 中两个新字段冗余无害，不需清理。

## Open Questions

- 来源权威性加权（spec 预留、初版全 0）：待积累 `evidence.round` 真实数据后，再决定是否按来源文档分级。
- 飞书 / graph_search 等低频工具的专属 chunker：初版走通用兜底，T4 按真实观测格式再细化。
