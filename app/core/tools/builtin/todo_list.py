# -*- coding: utf-8 -*-
"""文件所在目录：app/core/tools/builtin/todo_list.py
工具基类与任务规划管理工具的实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict
from pydantic import BaseModel

from app.core.tools import BaseTool


class ToolParameter(BaseModel):
    """JSON Schema 风格的单个参数描述（简化）。"""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True




class TodoItem(TypedDict):
    """单个待办事项的数据结构定义。"""

    content: str
    """任务的具体描述或内容。"""
    status: Literal["pending", "in_progress", "completed"]
    """任务当前所处的生命周期状态。"""


class AgentTodoPlannerTool(BaseTool):
    """Agent 任务规划与会话待办事项管理工具。

    继承自 BaseTool，用于让复杂任务背景下的 Agent 具备自我规划、动态调整步骤的能力。
    """

    name: str = "write_todos"

    # 1. 将原工具描述直接注入为类的说明文档属性
    description: str = (
        "Use this tool to create and manage a structured task list for your current work session. "
        "This helps you track progress and organize complex tasks. "
        "Only use this tool if you think it will be helpful in staying organized. If the user's request "
        "is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.\n\n"
        "## When to Use This Tool\n"
        "1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions.\n"
        "2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations.\n"
        "3. User explicitly requests todo list - When the user directly asks you to use the todo list.\n"
        "4. User provides multiple tasks - When users provide a list of things to be done.\n\n"
        "## Task States\n"
        "- pending: Task not yet started.\n"
        "- in_progress: Currently working on.\n"
        "- completed: Task finished successfully."
    )

    # 2. 将系统提示词定义为类常量，方便 Agent 框架层统一调用、拼装
    SYSTEM_PROMPT: str = (
        "## `write_todos` Tool Instructions\n"
        "You have access to the `write_todos` tool to help you manage and plan complex objectives.\n"
        "It is critical that you mark todos as completed as soon as you are done with a step. "
        "Do not batch up multiple steps before marking them as completed.\n"
        "When you finish all work, write your final answer in the message AFTER your last `write_todos` call. "
        "The user wants the final substance, not just a confirmation that the todo list is updated."
    )

    def __init__(self) -> None:
        """初始化工具参数。由于属于嵌套的复杂对象，参数描述将通过复写 Schema 方法来实现。"""
        super().__init__()
        # 兼容基类的设计，定义最外层的参数名称
        self.parameters = [
            ToolParameter(
                name="todos",
                type="array",
                description="The complete list of updated todo items.",
                required=True
            )
        ]

    def schema_parameters(self) -> dict[str, Any]:
        """复写基类方法，以支持大模型所需的严格嵌套的 Array[Object] JSON Schema 结构。

        Returns:
            符合标准 OpenAI Tools 规范的参数字典。
        """
        nested_properties: dict[str, Any] = {
            "content": {
                "type": "string",
                "description": "The description or text content of this specific task step."
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
                "description": "The operational status of the task item."
            }
        }

        schema_definition: dict[str, Any] = {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "An array containing all task items for the current session state.",
                    "items": {
                        "type": "object",
                        "properties": nested_properties,
                        "required": ["content", "status"]
                    }
                }
            },
            "required": ["todos"]
        }
        return schema_definition

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """执行待办事项的更新与清洗逻辑。

        Args:
            **kwargs: 大模型输入的参数字典，预期包含 'todos' 键。

        Returns:
            包含处理状态和规范化数据的工业级响应字典。
        """
        raw_todos: list[dict[str, Any]] = kwargs.get("todos", [])
        validated_todo_list: list[TodoItem] = []

        # 遍历并清洗数据，确保符合生产环境的健壮性要求
        for raw_item in raw_todos:
            cleaned_content: str = str(raw_item.get("content", "")).strip()
            raw_status: str = str(raw_item.get("status", "pending")).lower()

            # 状态守卫，防止非预期状态写入
            validated_status: Literal["pending", "in_progress", "completed"] = "pending"
            if raw_status in ["in_progress", "completed"]:
                validated_status = raw_status  # type: ignore[assignment]

            todo_node: TodoItem = {
                "content": cleaned_content,
                "status": validated_status
            }
            validated_todo_list.append(todo_node)

        # 组装结构明确的响应对象，不使用模糊的 'result'
        execution_response: dict[str, Any] = {
            "status": "success",
            "message": f"Successfully validated and updated {len(validated_todo_list)} planner tasks.",
            "data": {
                "todos": validated_todo_list
            }
        }

        return execution_response