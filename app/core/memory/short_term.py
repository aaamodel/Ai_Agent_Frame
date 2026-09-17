# -*- coding: utf-8 -*-
"""短期记忆：Redis 滑动窗口 + 超量时 LLM 摘要压缩。

改造（repo_map B1）：QwenChatLLMImpl 不再自己实例化 AsyncOpenAI + 直连 base_url，
改为内部持有 ``ModelRouter``，直接委托 ``ModelRouter.chat(messages=...)`` 返回
content；获得 Chat Tier（摘要场景=STANDARD 30s）+ 熔断降级 + 健康预选能力。
复杂度：调用层数保持 1（QwenChatLLMImpl.ainvoke → ModelRouter.chat），
与原 QwenChatLLMImpl.ainvoke → AsyncOpenAI.create 层数相同，**0 新增嵌套**。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Protocol, runtime_checkable

import tiktoken
from loguru import logger
from redis.exceptions import WatchError

from app.llm_model_router.model_router import ModelRouter
from app.models.agent_enums import MessageRole
from app.models.agent_schemas import Message


# ---------------------------------------------------------------------------
# Protocol（保持不变，用于 ShortTermMemory 运行期判定 LLM 是否合格）
# ---------------------------------------------------------------------------
@runtime_checkable
class CompressLLMProtocol(Protocol):
    """用于摘要压缩的 LLM：``async def ainvoke(input, **kwargs) -> str``。"""

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        ...


# =====================================================================
# 记忆压缩用 Chat 模型客户端（接入 ModelRouter 版本）
# =====================================================================
class QwenChatLLMImpl:
    """Qwen/千问 Chat 兼容模型摘要压缩客户端（通过 ModelRouter 调用）。

    与旧实现相比：
      ✅ 不再硬编码 base_url、不再持有独立 AsyncOpenAI 客户端；
      ✅ 享受 ModelRouter 的 Tier 路由、熔断降级、多模型切换；
      ✅ 对外 ``ainvoke`` 协议保持与旧实现完全一致（调用方零改动）。

    嵌套层数：1 层（QwenChatLLMImpl → ModelRouter.chat）
              与旧版（QwenChatLLMImpl → AsyncOpenAI.chat.completions.create）持平，
              属于 0 新增层接入。
    """

    def __init__(self, model_router: ModelRouter) -> None:
        """构造仅需一个全局 ModelRouter 单例。

        :param model_router: 来自 ``main._build_global_router()`` 的全局路由器。
        """
        if not isinstance(model_router, ModelRouter):
            raise TypeError(
                "QwenChatLLMImpl 现在需要一个 ModelRouter 实例，而不是 api_key。"
                "请在 main.py 用 _build_global_router() 构造后传入。"
            )
        self._model_router: ModelRouter = model_router

    async def ainvoke(self, input_prompt: str, **kwargs: Any) -> str:
        """实现 CompressLLMProtocol：输入 str prompt → 输出摘要 str。

        摘要场景命中 PURPOSE_TIER_MAP["chat"] = STANDARD 30s 超时。
        """
        messages = [{"role": "user", "content": str(input_prompt)}]
        try:
            resp = await self._model_router.chat(
                messages=messages,
                purpose_hint="chat",  # 声明场景 → STANDARD tier (30s)
                **kwargs,
            )
        except Exception as exc:
            logger.exception("QwenChatLLMImpl 摘要压缩（ModelRouter 路由）失败: {}", exc)
            # 与旧实现相同的降级策略：截断前 2000 字符，避免记忆压缩链路卡死
            return str(input_prompt)[:2000]

        content = getattr(resp, "content", None) or ""
        return str(content).strip() if isinstance(content, str) else str(content)


# =====================================================================
# 短期记忆主类（保持原逻辑不变：协议层检查仍然对 ainvoke 生效，因此
# ShortTermMemory 自己完全不用改，仅构造时传入的 llm 变为新版 QwenChatLLMImpl）
# =====================================================================
class ShortTermMemory:
    """短期记忆：基于 Redis 的滑动窗口 + 自动摘要压缩。"""

    def __init__(
        self,
        redis_client: Any,
        llm: Any,
        window_size: int = 20,
        max_tokens: int = 4000,
    ) -> None:
        """
        :param redis_client: ``redis.asyncio.Redis`` 实例
        :param llm: 用于摘要的模型，需实现 ``ainvoke``
        :param window_size: 最大保留消息条数（角色交替计一条）
        :param max_tokens: 触发按 token 压缩的阈值（近似）
        """
        self._redis = redis_client
        self._llm = llm
        self.window_size = window_size
        self.max_tokens = max_tokens
        self._key_prefix = "stm:session:"
        # 【方案 A】后台压缩任务强引用登记表（防 asyncio.create_task 协程被 GC 回收）
        self._background_compress_tasks: set = set()
        # 每个会话一把锁：串行化同一会话的后台压缩，避免并发重写同一 Redis 列表
        self._compress_locks: dict[str, asyncio.Lock] = {}
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def _count_tokens(self, text: str) -> int:
        """估算 token 数。"""
        if self._encoding is None:
            return max(1, len(text) // 4)
        return len(self._encoding.encode(text))

    def _serialize(self, msg: Message) -> str:
        payload = {
            "role": msg.role.value,
            "content": msg.content,
            "metadata": msg.metadata,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _deserialize(self, raw: str) -> Message:
        try:
            d = json.loads(raw)
            return Message(
                role=MessageRole(d["role"]),
                content=d["content"],
                metadata=d.get("metadata") or {},
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("反序列化消息失败，使用占位: {}", e)
            return Message(role=MessageRole.SYSTEM, content=raw, metadata={"error": "decode"})

    async def get_history(self, session_id: str) -> list[Message]:
        """读取会话全部消息（按时间顺序）。"""
        key = self._key(session_id)
        try:
            raw_list = await self._redis.lrange(key, 0, -1)
        except Exception as e:
            logger.exception("Redis LRANGE 失败: {}", e)
            raise RuntimeError(f"读取短期记忆失败: {e}") from e

        messages: list[Message] = []
        for raw_a in raw_list:
            if isinstance(raw_a, bytes):
                raw_a = raw_a.decode("utf-8", errors="replace")
            messages.append(self._deserialize(str(raw_a)))
        return messages

    async def clear(self, session_id: str) -> None:
        """清空指定会话的短期记忆。

        用途：会话重置、以及**评测隔离**——评测用一次性 session_id 跑完后清掉，
        否则下一轮评测会读到自己上一轮写的历史，测出来的 token/延迟会逐次膨胀。
        """
        try:
            await self._redis.delete(self._key(session_id))
        except Exception as e:
            logger.exception("清空短期记忆失败: {}", e)
            raise RuntimeError(f"清空短期记忆失败: {e}") from e

    async def add_message(
            self,
            session_id: str,
            message: Message,
            *,
            compress_async: bool = True,
    ) -> None:
        """追加一条消息并视需要触发压缩。

        RPUSH（毫秒级 Redis IO）始终同步完成，保证本条消息对本会话后续读取立即可见；
        超窗时的 LLM 摘要压缩（秒级）默认转入后台任务执行，不再阻塞对话主链路返回。

        :param compress_async: True（默认）超窗摘要后台化，仅触发即返回；
            False 保持同步压缩语义（供强一致/测试场景使用）。
        """
        key = self._key(session_id)
        try:
            await self._redis.rpush(key, self._serialize(message))
        except Exception as e:
            logger.exception("Redis RPUSH 失败: {}", e)
            raise RuntimeError(f"写入短期记忆失败: {e}") from e

        if compress_async:
            await self._maybe_spawn_background_compress(session_id)
        else:
            await self._compress_if_needed(session_id)

    # ------------------------------------------------------------------
    # 压缩判定与方案规划（同步 / 后台两条路径共享）
    # ------------------------------------------------------------------
    def _should_compress(self, messages: list[Message]) -> bool:
        """窗口溢出（条数超限或总 token 超限）且存在可压缩历史时返回 True。"""
        if not messages or len(messages) < 2:
            return False
        total_tokens = sum(self._count_tokens(m.content) for m in messages)
        return len(messages) > self.window_size or total_tokens > self.max_tokens

    def _plan_compression(
            self,
            messages: list[Message],
    ) -> Optional[tuple[list[Message], list[Message]]]:
        """规划压缩分区：返回 (待摘要区, 保留尾部)；不可压缩时返回 None。"""
        keep = max(2, self.window_size // 2)
        if keep >= len(messages):
            keep = len(messages) - 1
        if keep < 1:
            return None
        return messages[:-keep], messages[-keep:]

    async def _maybe_spawn_background_compress(self, session_id: str) -> None:
        """触发前轻量判定：超窗则把压缩任务提交到后台执行（快速返回，不等待 LLM 摘要）。"""
        messages = await self.get_history(session_id)
        if not self._should_compress(messages):
            return
        compress_task: asyncio.Task = asyncio.create_task(self._async_compress(session_id))
        self._background_compress_tasks.add(compress_task)
        compress_task.add_done_callback(self._on_background_compress_done)

    async def _async_compress(self, session_id: str) -> None:
        """后台压缩主流程：会话级互斥 → 双检是否需压缩 → LLM 摘要 → WATCH 原子替换。

        会话锁长驻（不清理），因为按需清理会引入新旧锁并存的并发重写窗口；
        锁数量与会话数成正比，内存占用可接受。
        """
        key = self._key(session_id)
        compress_lock: asyncio.Lock = self._compress_locks.setdefault(session_id, asyncio.Lock())
        async with compress_lock:
            messages = await self.get_history(session_id)
            if not self._should_compress(messages):
                return
            plan = self._plan_compression(messages)
            if plan is None:
                return
            to_summarize, tail = plan
            summary_text = await self._summarize_messages(to_summarize)
            summary_msg = Message(
                role=MessageRole.SYSTEM,
                content=f"[历史摘要]\n{summary_text}",
                metadata={"compressed_from": len(to_summarize)},
            )
            await self._rewrite_compressed(session_id, key, summary_msg, tail, baseline_len=len(messages))

    async def _rewrite_compressed(
            self,
            session_id: str,
            key: str,
            summary_msg: Message,
            tail: list[Message],
            baseline_len: int,
    ) -> bool:
        """基于 WATCH 乐观锁原子重写列表；期间有并发新写入则放弃本次压缩（不丢消息）。

        返回 True 表示重写成功；False 表示检测到并发写入已放弃（等待后续触发）。
        """
        for attempt_index in range(3):
            pipe = self._redis.pipeline()
            try:
                # 注意：WATCH 必须在 MULTI 之前发出，因此不能使用
                # `async with pipeline`（它会在进入时自动 MULTI），需显式管理。
                await pipe.watch(key)
                fresh_len = await pipe.llen(key)
                if fresh_len != baseline_len:
                    logger.info(
                        "会话 {} 压缩期间检测到新增消息，放弃本次压缩（等待后续写入触发）",
                        session_id,
                    )
                    return False
                pipe.multi()
                await pipe.delete(key)
                combined: list[Message] = [summary_msg, *tail]
                for message_item in combined:
                    await pipe.rpush(key, self._serialize(message_item))
                await pipe.execute()
                logger.info(
                    "会话 {} 已后台压缩，摘要 {} 条历史，保留 {} 条",
                    session_id,
                    baseline_len - len(tail),
                    len(tail),
                )
                return True
            except WatchError:
                logger.info("会话 {} 压缩重写乐观锁冲突（第 {} 次），重试", session_id, attempt_index + 1)
                continue
            finally:
                await pipe.reset()
        logger.warning("会话 {} 压缩重写连续冲突，已放弃本次后台压缩", session_id)
        return False

    def _on_background_compress_done(self, finished_task: asyncio.Task) -> None:
        """后台压缩任务完成回调：移出登记表并统一消费异常（不抛出）。"""
        self._background_compress_tasks.discard(finished_task)
        if finished_task.cancelled():
            return
        task_exception: Optional[BaseException] = finished_task.exception()
        if task_exception is not None:
            logger.warning("短期记忆后台压缩任务异常（已忽略）: {}", task_exception)

    async def _compress_if_needed(self, session_id: str) -> None:
        """同步压缩入口（compress_async=False 时使用，保持旧语义）。"""
        key = self._key(session_id)
        messages = await self.get_history(session_id)
        if not self._should_compress(messages):
            return
        plan = self._plan_compression(messages)
        if plan is None:
            return
        to_summarize, tail = plan
        summary_text = await self._summarize_messages(to_summarize)
        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[历史摘要]\n{summary_text}",
            metadata={"compressed_from": len(to_summarize)},
        )
        try:
            await self._redis.delete(key)
            combined = [summary_msg, *tail]
            for message_item in combined:
                await self._redis.rpush(key, self._serialize(message_item))
        except Exception as e:
            logger.exception("压缩重写 Redis 列表失败: {}", e)
            raise RuntimeError(f"记忆压缩失败: {e}") from e

        logger.info("会话 {} 已同步压缩，摘要 {} 条历史，保留 {} 条", session_id, len(to_summarize), len(tail))

    async def _summarize_messages(self, messages: list[Message]) -> str:
        """调用 LLM 生成摘要。"""
        lines = [f"{m.role.value}: {m.content}" for m in messages]
        shortmemory_summarize_prompt = (
            "请将以下对话压缩为简洁中文摘要，保留关键事实与用户意图：\n\n"
            + "\n".join(lines)
        )
        if not isinstance(self._llm, CompressLLMProtocol) and not callable(
            getattr(self._llm, "ainvoke", None)
        ):
            # 无可用 LLM 时退回截断
            return "\n".join(lines)[:2000]

        try:
            raw_content = await self._llm.ainvoke(shortmemory_summarize_prompt)
        except Exception as e:
            logger.exception("摘要 LLM 调用失败: {}", e)
            return "\n".join(lines)[:2000]

        if hasattr(raw_content, "content"):
            return str(getattr(raw_content, "content", "")).strip()
        if isinstance(raw_content, dict) and "content" in raw_content:
            return str(raw_content["content"]).strip()
        return str(raw_content).strip()
