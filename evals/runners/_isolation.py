# -*- coding: utf-8 -*-
"""评测记忆隔离：一次性 session_id + 独立记忆分区 + 跑完即清。

## 为什么必须隔离（这不是洁癖，是"指标可比性"问题）

``Orchestrator`` 每轮成功结束都会双写记忆：短期（Redis 滑动窗口）+
长期（Milvus 向量库），且**召回按 session_id 过滤**。评测 runner 如果复用
固定 session_id：

  1. 第 1 轮评测写下「问题+答案」；
  2. 第 2 轮评测同一个 session 召回第 1 轮的记录 → system prompt 变长 → token 变多；
  3. 第 N 轮变成"召回 N-1 条历史"，单轮 token **逐次膨胀**，延迟同步上涨。

结果就是：同一套代码连跑两次，第二次的 ``avg_tokens_per_turn`` 必然更高——
指标失去可比性，质量门形同虚设。

## 三条隔离规则

  1. **一次性 session_id**：每次 run 生成一个 tag，session_id 带上它
     （``eval::<runner>::<run_tag>::<case_id>``），保证跨 run 不重叠；
  2. **独立记忆分区**：统一 ``EVAL_SESSION_PREFIX`` 前缀，与线上真实 session
     天然隔离，也便于按前缀批量清理；
  3. **跑完即清**：每个用例结束后删掉该 session 的短期 Redis key 与 Milvus
     长期记忆行——"写进来但不留痕"，既测到了真实写路径，又不污染下一轮。

清理是 best-effort：失败只告警，不让评测因为清不动存储而中断（但会在
``cleanup_failures`` 里暴露出来，不静默吞掉）。
"""

from __future__ import annotations

from typing import Any, List
from uuid import uuid4

# 评测会话统一前缀（记忆分区标识，与线上真实 session_id 天然隔离）
EVAL_SESSION_PREFIX: str = "eval::"


def new_run_tag() -> str:
    """生成本次评测运行的唯一 tag（8 位十六进制）。"""
    return uuid4().hex[:8]


def build_eval_session_id(runner: str, case_id: str, run_tag: str) -> str:
    """构造一次性评测 session_id。

    Args:
        runner: runner 名（``tool`` / ``intent`` / ``rag``）。
        case_id: 黄金集用例 id。
        run_tag: 本次运行的唯一 tag（:func:`new_run_tag`）。

    Returns:
        形如 ``eval::tool::a1b2c3d4::T01`` 的隔离 session_id。
    """
    return f"{EVAL_SESSION_PREFIX}{runner}::{run_tag}::{case_id}"


async def purge_session(runtime: Any, session_id: str) -> bool:
    """清空指定 session 的短期（Redis）与长期（Milvus）记忆。

    Args:
        runtime: :class:`evals.runners._runtime.EvalRuntime`（须已构建 memory manager）。
        session_id: 要清理的隔离 session_id。

    Returns:
        True=清理成功（或本就没有记忆管理器）；False=发生异常（已记入日志）。
    """
    manager: Any = getattr(runtime, "_memory_manager", None)
    if manager is None:
        return True

    ok: bool = True
    short_term: Any = getattr(manager, "_stm", None)
    if short_term is not None and hasattr(short_term, "clear"):
        try:
            await short_term.clear(session_id)
        except Exception as exc:  # noqa: BLE001 - 清理失败不影响评测继续
            print(f"[eval-isolation] 短期记忆清理失败 {session_id}: {exc}", flush=True)
            ok = False

    long_term: Any = getattr(manager, "_ltm", None)
    if long_term is not None and hasattr(long_term, "forget_session"):
        try:
            await long_term.forget_session(session_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval-isolation] 长期记忆清理失败 {session_id}: {exc}", flush=True)
            ok = False
    return ok


async def purge_sessions(runtime: Any, session_ids: List[str]) -> int:
    """批量清理，返回清理失败的条数（供报告/日志暴露，不静默吞）。"""
    failures: int = 0
    for session_id in session_ids:
        if not await purge_session(runtime, session_id):
            failures += 1
    return failures


__all__ = [
    "EVAL_SESSION_PREFIX",
    "build_eval_session_id",
    "new_run_tag",
    "purge_session",
    "purge_sessions",
]
