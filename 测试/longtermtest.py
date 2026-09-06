# -*- coding: utf-8 -*-
"""本地测试脚本：使用通义千问向量验证长期记忆。"""
import os
import asyncio
from app.core.memory.long_term import LongTermMemory
from app.core.memory.long_term import QwenEmbeddingImpl, MilvusCollectionWrapper
from dotenv import load_dotenv
load_dotenv()

async def ceshi_run():
    # 1. 实例化百炼平台千问组件
    embed_model = QwenEmbeddingImpl(
        api_key=os.getenv("OPENAI_API_KEY"),  # 填入你的百炼 API Key
        model="text-embedding-v3"  # 默认 1024 维
    )

    # 2. 实例化 Milvus 包装器（注意 dim 改为 1024）
    milvus_coll = MilvusCollectionWrapper(
        collection_name="agent_qwen_ltm",
        dim=1024,  # 必须与 text-embedding-v3 的 1024 维保持一致
        host="127.0.0.1",
        port="19530"
    )

    # 3. 注入长期记忆管理器
    ltm_manager = LongTermMemory(milvus_collection=milvus_coll, embedding_model=embed_model)
    test_session = "session_entrepreneur_002"

    # 4. 测试存储
    print("\n--- 正在写入长期记忆 ---")
    mem_id = await ltm_manager.store(
        session_id=test_session,
        content="用户主营手机壳与B2B小商品供应链贸易，当前对东南亚跨境电商的物流账期很关心。",
        metadata={"category": "user_profile", "priority": "high"}
    )
    print(f"成功存入 Milvus，主键 ID 为: {mem_id}")

    # 5. 测试召回
    print("\n--- 正在进行语义召回 ---")
    user_query = "用户目前在供应链和出海方面有哪些关注的痛点？"
    memories = await ltm_manager.recall(query=user_query, session_id=test_session, top_k=2)

    print(f"针对提问 [{user_query}]，召回到的长期记忆有：")
    for item in memories:
        print(f" -> [距离分数: {item.score:.4f}] 记忆内容: {item.content}")


if __name__ == "__main__":
    asyncio.run(ceshi_run())