# Tasks: Evidence Board（证据板）

> 特性开关 `enable_evidence_board`（默认关）贯穿 T1–T3：关闭时三条发送视图保持现状，开关打开后启用证据板。每组任务完成后跑对应测试验证。

## 1. T1 核心管道（归一化 + 打分 + 去重 + 选择）

- [x] 1.1 创建 `app/core/agent/evidence/` 包与 `models.py`：定义 `EvidenceUnit`（uid/tool_name/round_idx/block_idx/text/source/ref/kind/score/simhash/selected/also_from/fetched）与轮次报告结构；验证：模块可导入，字段默认值单测通过
- [x] 1.2 在 state.py 新增 `evidence_units`（实现上改为**整体替换**而非 add：每轮要回写全量 selected/dup 标记）与 `evidence_meta`（整体替换，含 next_seq/queries/rounds）字段及初始值；验证：图初始 state 含两字段，既有状态机构建测试不回归
- [x] 1.3 实现 `chunkers.py`：切分器注册表、通用兜底切分器（双换行→中英文句边界滑窗重叠 1 句、MAX_UNIT_CHARS=600、SHORT_OBS_CHARS=200），error/status 识别；验证：短观测不切、超长块不丢首尾、错误串标记为 error 单测通过
- [x] 1.4 实现 RAG chunker（解析结果头与 `[n]` 块、来源文献、超长块句切）、SQL/报表 chunker（整表 table 单元、TABLE_MAX_ROWS=30 保留"共 N 行"）、file/grep chunker（路径/行块）；验证：各格式夹具切分结果与 source/ref 断言通过
- [x] 1.5 实现 `pipeline.py` 查询表示：jieba 分词 + 停用词表 + bigram，`Q=本轮query(1.0) ∪ 原始问题(0.6)`；验证：窄 query + 原始问题词命中的回归用例通过
- [x] 1.6 实现打分：BM25(k1=1.5,b=0.75) 归一 40 + 子串 20 + 实体 15 + 结构 10 + 新鲜度 15，阈值 S_MIN=35；验证：固定语料断言排序、实体命中、阈值落选取的单测通过
- [x] 1.7 实现去重：sha1 短路 + 64 位 SimHash 海明≤3 + 短中文 3-gram Jaccard≥0.85 且长度差<20%，also_from 合并；验证：改写句合并、多来源保留、不同内容不误杀单测通过
- [x] 1.8 实现选择：预算 `min(1200+400(round-1),2400)`、条数 `min(4+round,10)`、MMR(λ=0.7, 3-gram Jaccard)、句边界截断标注；验证：四轮预算曲线、top_k 5→10、同主题打散、截断标注、error 恒保留单测通过
- [x] 1.9 实现 `ingest_observation` 编排（切分→打分→去重→选择→轮次报告含 no_new_evidence）与 uid 递增（读 evidence_meta.next_seq）；验证：多轮连续入管的单元编号唯一、全判重/全低分置 no_new_evidence 单测通过

## 2. T2 三条发送视图接入与降级

- [x] 2.1 实现 `view.py` 单元属主渲染：在板（编号+来源+原文/截断标注）、判重（重复指针+另见来源）、低分（一行索引）、error/status（恒显一行）；验证：渲染快照单测通过，同一全文不出现两次
- [x] 2.2 实现聚合尾注（总量/上板/低相关/重复计数 + 回取提示 + no_new_evidence 换词提示）；验证：尾注内容与计数单测通过
- [x] 2.3 接入 FC：FC 循环工具观测后入管（随节点 update 写 state）；`_fc_assemble_messages` 按 tool_call_id 属主替换 tool 消息内容，尾注只挂最后一条 tool 消息；验证：tool 消息配对且 content 非空、system 与首轮 user 字节不变的单测通过
- [x] 2.4 接入文本协议：观测后入管，`compact_history_lines` 结果段切换属主渲染、尾注挂最后工具块；验证：文本协议消息夹具渲染通过
- [x] 2.5 接入 plan 提炼：L584 区域短观测原文直出、超长改为当前子任务入选单元 + "已省略"索引（无回取措辞），query=子任务描述+agent_goal+用户问题；error/empty 观测也入管；验证：尾部关键信息可见、plan 路径测试通过
- [x] 2.6 加特性开关 `enable_evidence_board`（默认关）：关闭时三视图走现状；验证：开关关闭时现有 383 测试全绿
- [x] 2.7 实现降级边界：ingest/view 异常 → warning + `evidence.fallback` 留痕 + 该轮回退现状压缩（FC 保留 compact_tool_observations 为 fallback）；验证：chunker 抛错注入测试中主流程不中断且视图正常生成

## 3. T3 fetch_evidence 按需回取

- [x] 3.1 实现 `fetch.py`：按 uid 定位单元；window=0 未截断返回 Unit.text，截断或 window>0 时经 ref（tool_call_id/round/block）从原 tool 消息重切取目标块与相邻块（window 0–5）；验证：纯本地取回单测通过
- [x] 3.2 两 ReAct 循环组装消息前去重追加 fetch_evidence function 定义（plan 不注入；文本协议以工具目录条目告知）；节点工具分发前拦截该调用，不进工具注册表、不触发 registry；验证：mock registry 断言回取零外部调用、plan 视图无回取提示
- [x] 3.3 取回单元标 fetched=True（ref.fetched_round=下一轮）：下轮选择 +10 优先且强制保留一轮，同 uid 重复回取不重复占预算；验证：取回后下轮视图保留、重复回取预算单测通过
- [x] 3.4 非法 uid / window 越界返回含可用编号的确定性错误文本，循环不中断；文本协议从动作行/JSON 解析回取指令复用同一拦截器；验证：两类协议的错误形态与继续执行单测通过

## 4. T4 可观测、夹具验收与收尾

- [x] 4.1 trace 埋点：`evidence.round`（new/duplicated/low/on_board/board_chars/no_new_evidence）、`evidence.fetch`（uid/window/chars/hit）、`evidence.fallback`（stage/error）；验证：跑一轮 FakeModel 会话后 trace 记录字段完整
- [x] 4.2 补全低频工具 chunker（网页/豆包/全网搜索按结果块、飞书/graph_search 结构化块，500 字上限）；验证：真实/夹具输出切分单测通过
- [x] 4.3 构造 3 轮 RAG（每轮 5 长片段）固定夹具：先在开关关闭时固化末轮发送视图 input token 基线，再断言开关打开后 ≤ 基线 40%；验证：夹具测试通过并记录前后数值（实测 2672→707 字符，26.5%）
- [x] 4.4 同一夹具标注 5 条关键证据 uid，断言末轮全部在板；端到端断言模型最终答案行为与关闭时一致；验证：召回与行为回归测试通过
- [x] 4.5 全量测试 `pytest 测试 -q` 不回归（基线 383 passed，新增用例后只增不减）；开关打开下复跑状态机与纠偏测试；验证：全量 405 全绿；开关开复跑状态机+纠偏 141 全绿
- [x] 4.6 删除 execute_node.py 中 `解析到了哪些事实`/`更新了哪些事实` 调试 print（如证据板工作完成时仍存在）；验证：grep 无残留、控制台无调试输出

## 5. T5 短观测豁免 + plan 延迟一步回取（v2 修订）

> 仍受 `enable_evidence_board` 开关控制；关闭时零行为变化。

- [x] 5.1 `models.py` 给 `EvidenceUnit` 加 `exempt: bool = False` 字段；`pipeline.py` 加常量 `EXEMPT_OBS_CHARS=300`、`EXEMPT_KEEP_ROUNDS=3`、`EXEMPT_STUB_HEAD_CHARS=60`；验证：字段默认值与常量单测通过
- [x] 5.2 `ingest_observation` 加豁免分支：整条非 error/status 观测 strip 后 ≤300 字符 → 跳过 chunker 建 1 个单元（exempt=True、score=0），仅做 sha1 精确判重（含跨豁免单元），命中置 dupe_of 且不占豁免名额；豁免单元不进打分/MMR/预算；验证：120 字观测建 1 单元且不打分、逐字重复观测判重、200–300 字分段观测仍整体豁免的单测通过
- [x] 5.3 ReAct 两视图（FC `render_fc_messages` / 文本协议属主渲染）加豁免渲染：属主豁免单元 `current_round-round_idx < 3` 原文直出且不占板预算；其后渲染桩 `[uid] 工具名 · 原始 N 字符 · 首句预览（可调取）`；尾注新增豁免原文数/豁免桩数计数；验证：第 0/1/2 轮原文、第 3 轮桩化、桩可经现有 fetch_evidence 取回、同一全文不出现两次的单测通过
- [x] 5.4 plan 视图 `render_plan_observation` 原文直出阈值 200 统一为 300（本步豁免单元原文呈现）；超长观测的落选索引措辞改为"已省略（低相关；如需全文，在 `requested_evidence_uids` 填入对应编号）"；验证：250 字观测原文直出、长观测省略索引新措辞快照单测通过
- [x] 5.5 `SubTaskOutcomeSchema` 加 `requested_evidence_uids: Optional[List[str]]`（max_length=3，默认 None）；distill 提示词加一条简短规则区分 `selected_alternative_id`（换资产/工具方向）与 `requested_evidence_uids`（索取本步工具观测全文），两者可同时留空或同时填写；验证：schema 解析与提示词渲染单测通过
- [x] 5.6 execute_node plan 路径实现延迟恢复：解析 distill 输出的 uids → 确定性校验（round_idx==cursor、content/table、确属本步省略集合，≤3 条，非法项拒绝并留痕）→ 在 cursor+1 插入 `correction_evidence_{subtask_id}` 内部步（伪工具 `evidence_restore`，不注册）→ 工具分发前拦截，按 uid 从 evidence_units/subtask_results 回填原文合成 observation（truncated 经 ref 重切，受 8000 字上限），零外部调用、回填内容不二次入管 → 恢复步正常 distill，其 requested_evidence_uids 被忽略（每子任务一次）；复用纠偏配额/留痕/链式闸门；验证：节点层单测（见 5.7）
- [x] 5.7 新增单测：受理 2 个合法 uid → 插入 1 个内部步且 registry 零调用、回填含两单元原文、其后再 distill；同时填两字段互不干扰；越步/伪造/error uid 被拒且结论正常应用；恢复步再次请求被忽略；恢复步 id 受链式闸门约束不再插纠偏；全量 `pytest 测试 -q` 不回归（现有 446 基线只增不减）
- [x] 5.8 trace 埋点：`evidence.round` 增 exempt 计数；新增 `evidence.restore`（task_id/uids/accepted/rejected/chars）；验证：FakeModel 会话中字段完整可断言
