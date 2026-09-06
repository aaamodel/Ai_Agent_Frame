# intent_utils.py
# ============================================================
# 工具类、装饰器、辅助函数
# ============================================================

import re
from typing import Any, Callable, Optional
from app.query_intent.intent_data_base import IntentResult  # 从核心模块导入 Result


# -------------------- 响应清理工具 --------------------
class LLMResponseCleaner:
    @staticmethod
    def strip_markdown_code_fence(raw: str) -> str:
        if not raw:
            return raw
        pattern = re.compile(r"```(?:\w+)?\n?([\s\S]*?)\n?```", re.MULTILINE)
        match = pattern.search(raw)
        if match:
            return match.group(1).strip()
        return raw.strip()



# -------------------- 统一结果工具 --------------------
class IntentSuccessResults:
    @staticmethod
    def success(data: Optional[Any] = None) -> IntentResult:
        return IntentResult(code=0, message="success", data=data)