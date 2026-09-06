# -*- coding: utf-8 -*-
from typing import Any
from loguru import logger
from app.core.tools.base import BaseTool, ToolParameter
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPBridgeTool(BaseTool):
    """
    通用 MCP 桥接适配器：
    把任何外界通过命令行（Node/Python）运行的标准 MCP 工具，无缝包装进你的 BaseTool 框架中。
    """

    def __init__(self, name: str, description: str, command: str, args: list[str], mcp_tool_name: str,
                 parameters: list[ToolParameter]):
        super().__init__()
        self.name = name
        self.description = description
        self.mcp_tool_name = mcp_tool_name  # 外界 MCP 服务器中实际声明的工具名
        self.parameters = parameters  # 手动声明映射给你当前 Agent 的参数规范

        # 配置标准输入输出的子进程启动参数
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )

    async def execute(self, **kwargs: Any) -> str:
        try:
            # 建立跨语言/跨进程的 stdio 管道连接
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 协议握手
                    await session.initialize()

                    # 核心代理：调用远端 MCP Server 中的对应工具
                    logger.info("正在通过 MCP 管道派发任务至远端工具: {}", self.mcp_tool_name)
                    mcp_response = await session.call_tool(self.mcp_tool_name, arguments=kwargs)

                    # 将远端返回的内容解析并拼接组合成纯文本返回给你的 Agent
                    text_outputs = [content.text for content in mcp_response.content if hasattr(content, 'text')]
                    return "\n".join(text_outputs) if text_outputs else "（执行完成，无文本输出）"

        except Exception as e:
            return f"MCP 桥接网关执行错误: {str(e)}"