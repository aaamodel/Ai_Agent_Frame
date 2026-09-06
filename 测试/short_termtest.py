# -*- coding: utf-8 -*-
"""本地测试脚本：验证 Redis 短期记忆与百炼 Qwen 自动压缩。"""

import asyncio
import os

import redis.asyncio as aioredis
from app.core.memory.short_term import ShortTermMemory ,QwenChatLLMImpl # 原脚手架核心
from app.models.agent_schemas import Message
from app.models.agent_enums import MessageRole
from dotenv import load_dotenv
load_dotenv()
async def run_short_term():
    # 1. 连上你本地 Docker 里的 Redis (默认 6379 端口)
    redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)

    # 2. 实例化百炼大模型打工人
    qwen_chat = QwenChatLLMImpl(
        api_key=os.getenv("OPENAI_API_KEY"),  # 换成你的百炼 API Key
        model="qwen-long"
    )

    # 3. 注入短期记忆管理器 (故意把窗口设为 3，方便看压缩效果)
    stm_manager = ShortTermMemory(
        redis_client=redis_client,
        llm=qwen_chat,
        window_size=3,  # 只要超过 3 条就开始压缩
        max_tokens=1000
    )

    session_id = "test_chat_session_999"

    # 清空之前的旧测试数据
    await redis_client.delete(f"stm:session:{session_id}")

    print("\n--- 1. 模拟前两轮正常对话 ---")
    dialogues = [
        Message(role=MessageRole.USER, content="你好，我是做跨境小商品B2B贸易的，主要做东南亚市场。"),
        Message(role=MessageRole.ASSISTANT,
                content="您好！非常高兴为您服务。针对东南亚跨境B2B贸易，我可以帮您分析物流账期、市场选品和供应链管理。"),
        Message(role=MessageRole.USER, content="我现在遇到一个痛点，马来西亚和越南的物流回款周期太长了，压款很严重。"),
    ]

    for msg in dialogues:
        await stm_manager.add_message(session_id, msg)
        print(f"追加消息成功 -> {msg.role.value}: {msg.content[:15]}...")

    # 打印当前数据库里的情况
    history = await stm_manager.get_history(session_id)
    print(f"\n当前 Redis 里的消息条数: {len(history)} 条（未触发压缩）")

    print("\n--- 2. 输入第 4 条消息，强行撑爆窗口（window_size=3），触发大模型压缩 ---")
    trigger_msg = Message(role=MessageRole.ASSISTANT,
                          content="明白，东南亚小规模外贸中，物流代收（COD）或长账期确实极其消耗现金流。我们可以考虑调整账期策略或选择P2P物流 arbitrage。")

    await stm_manager.add_message(session_id, trigger_msg)

    # 再次读取历史，见证奇迹的时刻
    compressed_history = await stm_manager.get_history(session_id)
    print(f"\n触发压缩后，当前 Redis 里的消息条数: {len(compressed_history)} 条")
    print("\n完整记忆结构回显：")
    for i, msg in enumerate(compressed_history):
        print(f"[{i}] {msg.role.value}: {msg.content}")

    await redis_client.close()


if __name__ == "__main__":
    asyncio.run(run_short_term())