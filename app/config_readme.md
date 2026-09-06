一、先理解「调用等级 → tier → 模型池」是怎么串起来的
目的到 tier 的映射写死在 infrastructure/llm/model_router.py:L68-L74：




planner        → STANDARD (30s 默认)
react          → FAST     (15s)
reflection     → DEEP     (60s)
intent_analysis→ FAST     (15s)   （query_intent 意图识别/改写）
chat           → STANDARD (30s)   （对话端点 / RAG / 短期记忆摘要）
也就是说，你把不同模型塞进"FAST 池 / STANDARD 池 / DEEP 池"，上面 5 种业务场景就会自动命中对应池，失败还会按候选顺序自动降级。

二、模型参数初始化入口（2 个）
入口 1：单模型兼容入口（你现在原来的方式，最简单，之前就有）
只配 .env 里 3 个基础变量就行：

env



OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_LLM_MODEL=qwen-plus
此时 3 个 tier 默认都复用这个模型，超时用硬编码默认值：FAST=15s / STANDARD=30s / DEEP=60s。

优点：零学习成本、和旧代码完全兼容。 缺点：所有等级都是同一个模型，体现不出"FAST 用便宜小模型、DEEP 用擅长思考的大模型"的收益。

入口 2：多模型 + 3 tier 差异化入口（本轮新加，完全用上了 fast / deep 等等级）
在 .env 里追加 7 个变量（字段含义都写在注释里了，直接复制改值）：

env



# =============== 多模型注册（JSON 数组字符串）=================
# 每个模型一个对象：
#   model_id          唯一 ID，必填；下面 LLM_TIER_* 用这个 ID 引用
#   api_key / base_url 省略 = 复用上面的 OPENAI_API_KEY / OPENAI_API_BASE
#   priority          同 tier 里的降级顺序（数字越小越先尝试）
#   supports_thinking 必须至少有 1 个=true 的模型出现在 DEEP tier；否则启动期直接报错拦截（Fail-Fast）
LLM_MODELS=[
  {"model_id":"qwen-turbo","priority":0,"supports_thinking":false},
  {"model_id":"qwen-plus", "priority":1,"supports_thinking":true},
  {"model_id":"qwen-long", "priority":2,"supports_thinking":true}
]

# =============== 每个 tier 的候选池（逗号分隔 or JSON 数组字符串都行）================
# 只给 qwen-turbo 跑 react 交互/意图识别（又快又省钱）
LLM_TIER_FAST=qwen-turbo
# 先跑 qwen-plus（质量好），失败降级到便宜的 qwen-turbo
LLM_TIER_STANDARD=qwen-plus,qwen-turbo
# 先用长上下文 thinking 模型 qwen-long 跑反思，失败降级到 qwen-plus；整个 tier 要求至少 1 个 supports_thinking=true
LLM_TIER_DEEP=qwen-long,qwen-plus

# =============== 每个 tier 的独立调用超时（毫秒）================
LLM_TIER_FAST_TIMEOUT_MS=8000     # 交互响应，用户不能等太久
LLM_TIER_STANDARD_TIMEOUT_MS=25000
LLM_TIER_DEEP_TIMEOUT_MS=90000     # 深度思考允许长一点
启动时日志会直接打印，一眼能核对是否生效： 构建全局 ModelRouter：注册 3 个模型，tier=[FAST:qwen-turbo/STANDARD:qwen-plus,qwen-turbo/DEEP:qwen-long,qwen-plus]，tier_timeout=[FAST:8000ms/STANDARD:25000ms/DEEP:90000ms]

三、Tier 配置真正生效的链路（这两点是代码里专门改的，避免"配了 tier 但实际上没用"）
Select 阶段：Selector 按 Purpose 解析 Tier（planner=STANDARD / react=FAST / reflection=DEEP），然后严格按你配的 candidates 顺序返回候选（已经由 v4 验证脚本 4 确认 3 条 Purpose 返回的顺序和超时 100% 对应上）。
Execute 阶段：之前代码里 timeout_ms 只是 Selector 存了字段、Validator 要求必填，但调用实际根本没套超时——现在已经在 async_model_executor.py:L86-L118 用 asyncio.wait_for(call, timeout=target.timeout_ms/1000) 把每一次候选调用都包了协程级超时。v4 验证第 6 项实测：FAST=200ms 配置，确实在 203ms 左右中断了 5 秒的 hang 调用，并且触发 mark_failure + 降级下一个候选（远小于 5 秒阻塞，证明 tier 超时是真的在工作，不是摆设）。
Fail-Fast 保护：如果你忘了给 DEEP 候选配 supports_thinking=true 的模型，应用启动就会直接抛 ValueError: deep-thinking-tier deep has no enabled candidate that supports thinking（v4 第 5 项复现通过），不会带着"坏配置上线 → reflection 静默降级到不 thinking 的模型"。
