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
    #: 取值约束：非空时以 JSON Schema ``enum`` 导出，把"自由文本参数"变成"受约束选择"。
    #: 承载的是**运行时实时清单**（知识库集合、图谱集合、工具名…），这些资产会随
    #: 上传/删除变化，因此不能静态写死——模型凭空编造名字的根因正是这里曾经无从表达约束。
    enum: list[Any] | None = None
    #: 数组元素的取值约束（JSON Schema ``items``）。
    #: ⚠️ ``type="array"`` 的取值域必须写在这里：把 ``enum`` 放在数组同一层，语义会变成
    #: "整个数组只能恰好等于这几个值之一"，而不是"数组元素只能从这几个值里取"。
    items: dict[str, Any] | None = None


class BaseTool(ABC):
    """所有 Agent 工具的抽象基类。"""

    name: str = "base_tool"
    description: str = "基础工具"

    def __init__(self) -> None:
        self.parameters: list[ToolParameter] = []

    def schema_parameters(self) -> dict[str, Any]:
        """导出为 OpenAI tools 风格的 parameters 结构。

        只有**显式设置**了 ``enum`` / ``items`` 的参数才会带上对应键——存量工具的导出
        结果必须与改动前逐字一致，不因新增能力而平白增加每次请求的 schema 体积。
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = list(p.enum)
            if p.items:
                prop["items"] = dict(p.items)
            properties[p.name] = prop
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

    复用工具的 schema_parameters()（参数为嵌套结构的工具会复写该方法），
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
