# intent_service.py
# ============================================================
# AI 服务接口
# ============================================================

from app.query_intent.intent_data_base import IntentChoiceTier, IntentChatRequest  # 从核心模块导入 Tier


class IntentLLMService:
    def chat(self, request: IntentChatRequest, tier: IntentChoiceTier) -> str:
        """
        调用 LLM 进行聊天，返回生成的文本。
        实际实现应由子类完成。
        """
        return ""