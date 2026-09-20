# -*- coding: utf-8 -*-
"""统一记忆管理：协调短期记忆与长期记忆。"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

from app.core.memory.long_term import LongTermMemory
from app.core.memory.short_term import ShortTermMemory
from app.models.agent_enums import MessageRole
from app.models.agent_schemas import MemoryContext, Message


class MemoryManager:
    """
    统一记忆管理器：协调短期记忆和长期记忆。
    完美实现 app/core/agent/orchestrator.py 中的 MemoryManager(Protocol) 契约。
    """

    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory) -> None:
        """
        :param short_term: 短期记忆实现（Redis + 窗口）
        :param long_term: 长期记忆实现（向量库）
        """
        self._stm = short_term
        self._ltm = long_term

    async def forget_session(self, session_id: str) -> Dict[str, str]:
        """删除某会话在后端的全部记忆（短期 Redis + 长期向量库）。

        用途：用户在界面上删除会话时，必须把后端按 ``session_id`` 存的东西
        一起清掉，否则就是"删了还在"——下一轮召回仍会命中旧记录。

        ⚠️ 两个存储**分开 try**：一个失败不能阻止另一个被清。
        返回逐项结果（``ok`` / ``failed: ...``）供接口如实回传，让"只清掉一半"
        在界面上可见——谎报成功会让用户以为数据已经没了。
        """
        outcome: Dict[str, str] = {}

        try:
            await self._stm.clear(session_id)
            outcome["short_term"] = "ok"
        except Exception as stm_error:  # noqa: BLE001 - 单侧失败不阻断另一侧
            logger.exception("删除会话短期记忆失败: session_id={}", session_id)
            outcome["short_term"] = f"failed: {stm_error}"

        try:
            await self._ltm.forget_session(session_id)
            outcome["long_term"] = "ok"
        except Exception as ltm_error:  # noqa: BLE001 - 同上
            logger.exception("删除会话长期记忆失败: session_id={}", session_id)
            outcome["long_term"] = f"failed: {ltm_error}"

        return outcome

    async def get_context(self, session_id: str, query: str, limit: int = 6) -> MemoryContext:
        """获取与当前查询相关的记忆上下文（短期历史 + 长期召回）。

        短期历史（Redis 低延迟）与长期召回（embedding + Milvus，内部已
        ``asyncio.to_thread`` 化，事件循环不被阻塞）通过 ``asyncio.gather``
        并行发起：两路墙钟等待由串行的 sum 降为 max，缩短记忆装载总耗时。
        """
        async def _load_short_history() -> list[Any]:
            try:
                return await self._stm.get_history(session_id)
            except Exception as e:
                logger.exception("读取短期记忆失败: {}", e)
                return []

        async def _load_long_recall() -> list[Any]:
            try:
                # 💡 将外层传进来的 limit 动态赋值给长期记忆的 top_k
                return await self._ltm.recall(query, session_id, top_k=limit)
            except Exception as e:
                logger.exception("长期记忆召回失败: {}", e)
                return []

        short_msgs, long_items = await asyncio.gather(
            _load_short_history(),
            _load_long_recall(),
        )

        return MemoryContext(
            session_id=session_id,
            short_term_messages=short_msgs,
            long_term_items=long_items,
        )

    async def save(self, session_id: str, message: Message) -> None:
        """将新消息写入短期记忆（滑动窗口与压缩由 ShortTermMemory 负责）。"""
        try:
            await self._stm.add_message(session_id, message)
        except Exception as e:
            logger.exception("保存短期记忆失败: {}", e)
            raise RuntimeError(f"save 失败: {e}") from e

    # ---------------------------------------------------------------------------
    # 精准实现 Orchestrator 的 Protocol 契约方法
    # ---------------------------------------------------------------------------

    async def get_relevant(self, session_id: str, query: str, limit: int = 8) -> List[str]:
        """
        根据用户当前输入，从长期记忆库（向量库）中召回最相关的历史背景碎片。
        :param session_id: 会话 ID
        :param query: 用户的当前提问
        :param limit: 限制召回条数
        :return: 文本片段列表（给编排器直接注入到 Prompt 中作为背景知识）
        """
        try:
            # 调度之前通关的向量长期记忆 recall 方法
            long_items = await self._ltm.recall(query, session_id, top_k=limit)

            # 编排层需要的是纯文本列表，因此我们把 LongMemoryItem 里的文本内容提取出来
            # 备注：由于具体 LongMemoryItem 字段可能为 content 或 text，这里根据你的具体模型按需读取
            relevant_texts: List[str] = []
            for item in long_items:
                if hasattr(item, "content"):
                    relevant_texts.append(str(item.content))
                elif hasattr(item, "text"):
                    relevant_texts.append(str(item.text))
                else:
                    relevant_texts.append(str(item))
            return relevant_texts

        except Exception as e:
            logger.exception("编排层调用长期记忆 get_relevant 失败: {}", e)
            return []

    async def append_turn(
            self,
            session_id: str,
            role: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        由编排层直接调用的对话流追加入口。
        :param role: 角色字符串（如 "user", "assistant", "system"）
        :param content: 消息文本内容
        :param metadata: 附加元数据
        """
        try:
            # 1. 将散装的字符串角色转化为系统底层的 MessageRole 枚举
            try:
                role_enum = MessageRole(role.lower())
            except ValueError:
                # 安全降级：如果编排层传入了无法识别的角色字符串，默认转为 SYSTEM 角色
                role_enum = MessageRole.SYSTEM

            # 2. 组装为内部标准 Message 模型
            message = Message(
                role=role_enum,
                content=content,
                metadata=metadata or {}
            )

            # 3. 调度已有的 save 方法，存入 Redis
            await self.save(session_id, message)
            logger.debug("编排层单轮对话成功追加至短期记忆: session={}, role={}", session_id, role)

        except Exception as e:
            logger.exception("编排层调用 append_turn 失败: {}", e)
            raise RuntimeError(f"append_turn 失败: {e}") from e