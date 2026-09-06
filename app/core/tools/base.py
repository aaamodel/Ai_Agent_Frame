# -*- coding: utf-8 -*-
"""文件所在目录：app/core/tools/base.py
工具基类：统一 name、description、parameters 与 execute 接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """JSON Schema 风格的单个参数描述（简化）。"""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class BaseTool(ABC):
    """所有 Agent 工具的抽象基类。"""

    name: str = "base_tool"
    description: str = "基础工具"

    def __init__(self) -> None:
        self.parameters: list[ToolParameter] = []

    def schema_parameters(self) -> dict[str, Any]:
        """导出为 OpenAI tools 风格的 parameters 结构。"""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {"type": "object", "properties": properties, "required": required}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """执行工具逻辑；子类实现具体行为。"""


def tool_to_function_call_definition(
    tool: BaseTool,
    include_system_prompt: bool = True,
) -> dict[str, Any]:
    """把 BaseTool 转为 OpenAI tools[] 单条 function 定义（React / Planner 强制取参共用）。

    复用工具的 schema_parameters()（write_todos 等特化工具已复写为嵌套结构），
    并把工具的 SYSTEM_PROMPT 专属守则拼入 description，确保模型拿到足够说明。

    Args:
        tool: 已注册的具体工具实例
        include_system_prompt: 是否把工具的 SYSTEM_PROMPT 追加进 description

    Returns:
        可直接放入 OpenAI ``tools`` 数组的定义 dict：
        {"type": "function", "function": {name, description, parameters}}
    """
    description_parts: list[str] = [str(getattr(tool, "description", "") or "").strip()]
    if include_system_prompt:
        system_prompt: Any = getattr(tool, "SYSTEM_PROMPT", None)
        if system_prompt:
            description_parts.append(str(system_prompt).strip())
    description: str = "\n\n".join(part for part in description_parts if part) or str(tool.name)

    parameters: dict[str, Any] = tool.schema_parameters()
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        # schema_parameters 异常兜底：至少给空对象结构，避免触发 SDK 参数校验失败
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description,
            "parameters": parameters,
        },
    }
