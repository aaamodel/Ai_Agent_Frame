# -*- coding: utf-8 -*-
"""测试 Tracer 的运行轨迹与打印树状结构"""

import asyncio
import time
from app.infrastructure.trace.tracer import Tracer


async def simulate_agent_run():
    # 1. 初始化追踪器
    tracer = Tracer()

    # 2. 模拟用户发起了一次请求（生成单次聊天的物流单号 trace_id）
    # 假设此时用户的 session_id 是 "customer_zhang_999"
    trace_id = tracer.new_trace_id()
    print(f"🚀 [用户发送消息] 分配全新 trace_id: {trace_id}\n")

    # =========================================================================
    # 步骤一：Agent 整体接收到任务，开启总环节（Root Span）
    # =========================================================================
    root_span = tracer.start_span(
        name="Agent:处理跨境物流退税与利润查询",
        trace_id=trace_id,
        attributes={"session_id": "customer_zhang_999"}
    )
    # 模拟系统记录了一个突发事件日志
    tracer.log_event(trace_id, name="agent_start", payload={"input": "查上月手机壳退税"})
    await asyncio.sleep(0.1)  # 模拟微小耗时

    # =========================================================================
    # 步骤二：Agent 决定去查数据库（分裂出第一个子环节 Child Span）
    # =========================================================================
    db_span = tracer.start_span(name="Tool:查询PostgreSQL数据库", trace_id=trace_id)
    try:
        print(" -> [执行中] 正在拼命读取数据库...")
        await asyncio.sleep(0.3)  # 模拟查数据库耗时 300ms
        # 假设查到了数据，完工
        tracer.end_span(db_span)
    except Exception as e:
        tracer.end_span(db_span, error=e)

    # =========================================================================
    # 步骤三：Agent 拿着数据去调大模型（分裂出第二个子环节 Child Span）
    # =========================================================================
    llm_span = tracer.start_span(
        name="LLM:调用通义千问进行利润深度计算",
        trace_id=trace_id,
        attributes={"model": "qwen-max"}
    )
    try:
        print(" -> [执行中] 正在等待大模型推理吐字...")
        await asyncio.sleep(0.5)  # 模拟大模型思考耗时 500ms

        # 故意模拟一个偶发错误：比如大模型调用超时了
        raise TimeoutError("通义千问 API 响应超时(504)")
    except Exception as e:
        # 记录报错完工
        tracer.end_span(llm_span, error=e)

    # =========================================================================
    # 步骤四：大模型挂了，Agent 触发自动降级，调用备份模型（第三个子环节）
    # =========================================================================
    llm_backup_span = tracer.start_span(
        name="LLM:自动降级调用 DeepSeek 备份模型",
        trace_id=trace_id,
        attributes={"model": "deepseek-chat"}
    )
    print(" -> [执行中] 触发故障降级，正在请求备份模型...")
    await asyncio.sleep(0.4)  # 模拟备份模型耗时 400ms
    tracer.end_span(llm_backup_span)

    # =========================================================================
    # 步骤五：总环节收尾
    # =========================================================================
    tracer.log_event(trace_id, name="agent_success", payload={"output": "利润计算完成"})
    tracer.end_span(root_span)

    # =========================================================================
    # 📊 终极揭秘：看看数据在内存里被记账成了什么样子？
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 TRACER 内存全链路记账本账单树状复盘")
    print("=" * 60)

    # 凭 trace_id 把刚才所有的历史账单捞出来
    record = tracer.get_trace(trace_id)

    if record:
        print(f"物流总单号 (trace_id): {record.trace_id}")
        print(f"事件流水账 (events count): {len(record.events)}")
        for ev in record.events:
            print(f"  🎈 [事件占印] {ev['event_name']} -> {ev['payload']}")

        print("\n环节时序树 (Spans):")
        for i, span in enumerate(record.spans):
            # 判断是不是根节点
            prefix = "  ┗━━ 根节点:" if span.parent_span_id is None else "      ┗━━ 子环节:"
            status = "❌ 失败" if span.error else "✅ 成功"

            # 计算耗时（由于用的是 perf_counter，我们直接看相对耗时）
            duration = span.end_time - span.start_time if span.end_time else 0

            print(f"{prefix} [{span.operation}]")
            print(f"          环节ID: {span.span_id}")
            print(f"          父级ID: {span.parent_span_id}")
            print(f"          耗时: {duration:.4f} 秒 | 状态: {status}")
            if span.error:
                print(f"          ⚠️ 报错详情: {span.error}")
            if span.attributes:
                print(f"          ⚙️ 附加属性: {span.attributes}")
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(simulate_agent_run())