# -*- coding: utf-8 -*-
"""工具注册中心：集中管理可用工具，生成 Prompt 描述，并处理编排层的调度执行。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.tools.base import BaseTool

# Langfuse 可观测性：@observe 在未配置密钥时自动退化为 no-op（零侵入）
from langfuse import observe as langfuse_observe


class ToolRegistry:
    """
    工具注册中心：管理所有可用工具。
    完美实现 app/core/agent/orchestrator.py 中的 ToolRegistry(Protocol) 契约。
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册工具；同名覆盖并记录日志。"""
        if tool.name in self._tools:
            logger.warning("工具 [{}] 已存在，将被覆盖", tool.name)
        self._tools[tool.name] = tool
        logger.info("已注册工具: {}", tool.name)

    def get_tool(self, name: str) -> BaseTool:
        """按名称获取工具。"""
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        return self._tools[name]

    def get_all_tools(self) -> List[BaseTool]:
        """返回全部工具列表。"""
        return list(self._tools.values())

    def get_tools_description(self) -> str:
        """生成所有工具的自然语言描述（用于 System Prompt）。"""
        lines: list[str] = []
        for t in self._tools.values():
            params = ", ".join(f"{p.name}: {p.type}" for p in t.parameters) or "无"
            lines.append(f"- {t.name}: {t.description}（参数: {params}）")
        return "\n".join(lines) if lines else "（当前无可用工具）"

    # ---------------------------------------------------------------------------
    # 精准实现 Orchestrator 的 Protocol 契约方法
    # ---------------------------------------------------------------------------

    def list_tool_names(self) -> List[str]:
        """
        列出所有已注册的工具名称。
        给编排器用来与 IntentContext.allowed_tools 做交集过滤。
        """
        return list(self._tools.keys())

    @langfuse_observe(name="tool_invoke", as_type="tool", capture_input=False, capture_output=False)
    async def invoke(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        根据工具名称和参数，异步调度执行具体工具。
        :param name: 工具注册名 (如 'database_query')
        :param arguments: 大模型解析出来的参数字典 (如 {'sql': 'SELECT...'})
        :return: 工具运行后的统一文本结果（异常时捕获并返回错误提示，确保 Agent 不崩溃）
        """
        try:
            # 1. 捞出对应的工具打工人
            tool = self.get_tool(name)
            logger.info("Orchestrator 正在调用工具 [{}]，参数: {}", name, arguments)

            # 2. 异步执行工具
            # 因为所有的具体工具（如 DatabaseQueryTool）都继承了 BaseTool，
            # 它们的 execute 方法都是 async def，所以这里直接 await
            result = await tool.execute(**arguments)

            # 3. 统一转化为字符串返回给大模型
            return str(result)

        except KeyError:
            error_msg = f"错误：试图调用未注册的工具 [{name}]。"
            logger.error(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"工具 [{name}] 执行期间发生异常: {str(e)}"
            logger.exception(error_msg)
            return error_msg


# =============================================================================
# 工具调用预算（单次 Agent 请求作用域）：质量感知的动态熔断
# =============================================================================
@dataclass
class ToolCallBudget:
    """单次 Agent 运行内的工具调用预算（质量感知，先检查后计数）。

    三层约束（互不替代，谁先触达谁生效）：
      1. 单工具总调用硬上限 ``per_tool_limits[name]``（缺省 10）：只要返回有用数据就允许调用，
         用完即熔断（区别于旧的"固定 3 次"）。
      2. 累计无效次数 ``invalid_limit``（缺省 3）：返回空 / 异常 / 系统错误提示等无效结果
         累计达 3 次 → 该工具硬熔断（成功不冲销）。
      3. 相关性抽查（``relevance_check_call``，缺省 5）：单个工具第 5 次调用结果若与
         调用意图完全不匹配（由 LLM 判定，如"搜书籍返回股票"）→ 该工具硬熔断。
     另有全局总调用硬上限 ``total_budget``（缺省 20）作为兜底总闸。

    熔断后的工具进入 ``_locked``，后续 ``can_call`` 一律拒绝，Observation 会把"不可再调用"
    的状态明确告知大模型。
    """

    per_tool_limits: Dict[str, int]
    """单工具总调用硬上限（name -> 允许的总次数）。"""
    default_per_tool: int
    """未单独配置时的单工具总调用上限（缺省 10）。"""
    total_budget: int
    """全局总调用硬上限（缺省 20；<=0 表示不限总数）。"""
    invalid_limit: int = 3
    """累计无效结果上限（空/异常/系统错误），达到即熔断该工具。"""
    relevance_check_call: int = 5
    """相关性抽查触发点：单个工具第 N 次调用结果做一次 LLM 相关性判定。"""
    _used_per_tool: Dict[str, int] = field(default_factory=dict)
    """各工具已发生的总调用次数（含无效调用）。"""
    _invalid_per_tool: Dict[str, int] = field(default_factory=dict)
    """各工具累计无效结果次数（成功调用不冲销）。"""
    _locked: Dict[str, str] = field(default_factory=dict)
    """已熔断工具及其原因（tool_name -> reason）。"""
    _total_used: int = 0
    """全局总调用次数。"""

    # ------------------------------------------------------------------
    # 限额 / 余量查询
    # ------------------------------------------------------------------
    def limit_of(self, tool_name: str) -> int:
        """返回指定工具的单次总调用硬上限（缺省返回全局默认上限）。"""
        return int(self.per_tool_limits.get(tool_name, self.default_per_tool))

    def used_of(self, tool_name: str) -> int:
        """返回指定工具当前已用总次数。"""
        return int(self._used_per_tool.get(tool_name, 0))

    def invalid_of(self, tool_name: str) -> int:
        """返回指定工具当前累计无效次数。"""
        return int(self._invalid_per_tool.get(tool_name, 0))

    def is_locked(self, tool_name: str) -> bool:
        """该工具是否已被熔断（不可再调用）。"""
        return tool_name in self._locked

    def lock_reason(self, tool_name: str) -> str:
        """返回熔断原因文案；未熔断返回空串。"""
        return str(self._locked.get(tool_name, ""))

    def total_remaining(self) -> int:
        """返回全局剩余额度（total_budget<=0 视为无上限，返回 -1）。"""
        return -1 if self.total_budget <= 0 else max(0, self.total_budget - self._total_used)

    # ------------------------------------------------------------------
    # 熔断判定（先检查后计数）
    # ------------------------------------------------------------------
    def can_call(self, tool_name: str) -> bool:
        """调用前判定：已熔断 / 单工具上限 / 全局上限任一触达则拒绝。"""
        if self.is_locked(tool_name):
            return False
        if self.used_of(tool_name) >= self.limit_of(tool_name):
            return False
        if self.total_budget > 0 and self._total_used >= self.total_budget:
            return False
        return True

    def consume(self, tool_name: str) -> bool:
        """尝试消费一次额度；成功则计数并返回 True，否则返回 False。"""
        if not self.can_call(tool_name):
            return False
        self._used_per_tool[tool_name] = self.used_of(tool_name) + 1
        self._total_used += 1
        return True

    def reached_relevance_checkpoint(self, tool_name: str) -> bool:
        """是否恰好到达"需要做一次 LLM 相关性抽查"的第 N 次调用。"""
        return (
            not self.is_locked(tool_name)
            and self.used_of(tool_name) == self.relevance_check_call
        )

    def record_invalid(self, tool_name: str) -> bool:
        """登记一次无效结果（空/异常/系统错误提示）。

        Returns:
            本次登记是否恰好触发熔断（累计无效达到 invalid_limit）。
        """
        if self.is_locked(tool_name):
            return False
        new_invalid: int = self.invalid_of(tool_name) + 1
        self._invalid_per_tool[tool_name] = new_invalid
        if new_invalid >= self.invalid_limit:
            self.lock_tool(tool_name, "invalid")
            return True
        return False

    def lock_tool(self, tool_name: str, reason: str) -> None:
        """按原因硬熔断某工具（重复熔断保留首次原因）。"""
        if tool_name not in self._locked:
            self._locked[tool_name] = reason
            logger.warning(
                "工具 [%s] 触发硬熔断，原因=%s（已用 %d/%d，累计无效 %d/%d）",
                tool_name, reason, self.used_of(tool_name), self.limit_of(tool_name),
                self.invalid_of(tool_name), self.invalid_limit,
            )

    def _reason_text(self, tool_name: str) -> str:
        """把熔断原因/触达条件转成对人类与 LLM 都清晰的中文说明。"""
        if self.is_locked(tool_name):
            reason: str = self.lock_reason(tool_name)
            if reason == "invalid":
                return f"工具 [{tool_name}] 累计返回 {self.invalid_limit} 次无效结果（空数据/异常/系统错误），已硬熔断"
            if reason == "relevance":
                return f"工具 [{tool_name}] 第 {self.relevance_check_call} 次调用结果与调用意图完全不匹配，已硬熔断"
            return f"工具 [{tool_name}] 已被系统熔断（{reason}）"
        used_limit = self.limit_of(tool_name)
        if self.used_of(tool_name) >= used_limit:
            return f"工具 [{tool_name}] 已达到单工具总调用上限（{used_limit} 次）"
        if self.total_budget > 0 and self._total_used >= self.total_budget:
            return f"本次任务总工具调用额度（{self.total_budget} 次）已全部用完"
        return f"工具 [{tool_name}] 当前不可调用"

    def deny_text(self, tool_name: str) -> str:
        """构造熔断 Observation：说明原因并引导模型转向其它工具或直接作答。"""
        remaining = self.total_remaining()
        extra_hint = ""
        if self.total_budget > 0:
            extra_hint = (
                "本次任务已无工具调用额度" if remaining == 0
                else f"剩余可用额度 {remaining} 次"
            )
        return (
            f"【系统拒绝执行】：{self._reason_text(tool_name)}。\n"
            "请绝对不要再次尝试调用该工具！请结合已有信息直接回答，或转向其它仍有余量且未被熔断的工具。"
            + (f"\n（{extra_hint}）" if extra_hint else "")
        )

    def lock_notice(self, tool_name: str) -> str:
        """工具刚刚被熔断时，追加在返回内容前的状态说明（让模型立刻知道该工具不可再用）。"""
        return f"【工具状态】{self._reason_text(tool_name)}，后续禁止再调用。\n"

    # ------------------------------------------------------------------
    # 供 LLM 动态规划的额度提示
    # ------------------------------------------------------------------
    def snapshot_lines(self) -> List[str]:
        """额度快照行列表（含熔断状态 / 累计无效数）。"""
        lines: List[str] = []
        if self.total_budget > 0:
            lines.append(f"- 总体工具调用额度：已用 {self._total_used} / 上限 {self.total_budget}")
        else:
            lines.append(f"- 总体工具调用额度：已用 {self._total_used} / 无硬上限")
        for name in sorted(self.per_tool_limits):
            if self.is_locked(name):
                lines.append(f"- {name}：已熔断（{self.lock_reason(name)}），禁止再调用")
            else:
                lines.append(
                    f"- {name}：已用 {self.used_of(name)}/{self.limit_of(name)}，"
                    f"累计无效 {self.invalid_of(name)}/{self.invalid_limit}"
                )
        return lines

    def live_prompt(self) -> str:
        """生成注入 LLM 消息的预算提示段（动态规划用，随已用量实时变化）。"""
        lines = self.snapshot_lines()
        if not lines:
            return ""
        return (
            "## 当前工具调用额度（请据此动态规划后续步骤）\n"
            + "\n".join(lines)
            + "\n已熔断/达上限的工具禁止再次调用；若总额度用完，请直接结合已有信息给出最终答案。"
        )

    def status_suffix(self, tool_name: str) -> str:
        """单次工具执行成功后拼接在 Observation 末尾的额度状态行（轻量提醒）。"""
        remaining = self.total_remaining()
        total_note = f"，总剩余 {remaining}" if self.total_budget > 0 else ""
        locked_note = "（已熔断）" if self.is_locked(tool_name) else ""
        return (
            f"[额度状态] {tool_name} 已用 {self.used_of(tool_name)}/{self.limit_of(tool_name)}"
            f"，累计无效 {self.invalid_of(tool_name)}/{self.invalid_limit}{total_note}{locked_note}"
        )


def build_tool_call_budget(
    config: Any,
    tool_registry: Any,
    tool_names: List[str],
) -> ToolCallBudget:
    """按优先级聚合各工具的调用上限并构建（质量感知的）预算对象。

    单工具总调用上限优先级（高 → 低）：
      1. 工具注册实例类属性 ``max_calls``（开发者显式声明，如 file_read_tool=10）；
      2. 编排配置 ``tool_max_attempts_<tool_name>``（运维按工具名覆写）；
      3. 全局默认 ``tool_max_attempts``（缺省 10，环境变量 TOOL_MAX_ATTEMPTS）。

    其它预算参数：
      - 全局总硬上限 ``tool_max_total_calls``（缺省 20，环境变量 TOOL_MAX_TOTAL_CALLS），
        <=0 表示不限制总数；
      - 累计无效次数上限 ``tool_invalid_attempts``（缺省 3，环境变量 TOOL_INVALID_ATTEMPTS）；
      - 相关性抽查触发点 ``tool_relevance_check_call``（缺省 5，环境变量 TOOL_RELEVANCE_CHECK_CALL）。

    Args:
        config: 编排配置（具备 .get(key, default) 的对象，如 dict）
        tool_registry: ToolRegistry（需可解析出工具实例以读取 max_calls）
        tool_names: 本次请求允许调用的工具名列表

    Returns:
        ToolCallBudget 实例
    """

    def _first_int(*values: Any, fallback: int) -> int:
        """取第一个能转成 >0 int 的值；全无效回退 fallback。"""
        for value in values:
            if value is None:
                continue
            try:
                parsed: int = int(value)
            except (TypeError, ValueError):
                continue
            return max(0, parsed)
        return fallback

    default_per_tool_raw: Any = (
        config.get("tool_max_attempts", os.getenv("TOOL_MAX_ATTEMPTS"))
        if hasattr(config, "get")
        else os.getenv("TOOL_MAX_ATTEMPTS")
    )
    default_per_tool: int = _first_int(default_per_tool_raw, fallback=10)

    total_raw: Any = (
        config.get("tool_max_total_calls", os.getenv("TOOL_MAX_TOTAL_CALLS"))
        if hasattr(config, "get")
        else os.getenv("TOOL_MAX_TOTAL_CALLS")
    )
    total_budget: int = _first_int(total_raw, fallback=20)

    invalid_raw: Any = (
        config.get("tool_invalid_attempts", os.getenv("TOOL_INVALID_ATTEMPTS"))
        if hasattr(config, "get")
        else os.getenv("TOOL_INVALID_ATTEMPTS")
    )
    invalid_limit: int = max(1, _first_int(invalid_raw, fallback=3))

    checkpoint_raw: Any = (
        config.get("tool_relevance_check_call", os.getenv("TOOL_RELEVANCE_CHECK_CALL"))
        if hasattr(config, "get")
        else os.getenv("TOOL_RELEVANCE_CHECK_CALL")
    )
    relevance_check_call: int = max(1, _first_int(checkpoint_raw, fallback=5))

    per_tool_limits: Dict[str, int] = {}
    for name in tool_names:
        limit_value: Any = None

        # 1) 工具实例类属性 max_calls（开发者声明）
        tool_instance: Optional[Any] = None
        try:
            if hasattr(tool_registry, "get_tool"):
                tool_instance = tool_registry.get_tool(name)
            else:
                tool_instance = getattr(tool_registry, "_tools", {}).get(name)
        except Exception:  # noqa: BLE001 - 工具缺失等异常一律按缺省处理
            tool_instance = None
        if tool_instance is not None:
            declared_limit: Any = getattr(tool_instance, "max_calls", None)
            if declared_limit is not None:
                limit_value = declared_limit

        # 2) 运维按工具名覆写（tool_max_attempts_<tool_name>）
        if limit_value is None and hasattr(config, "get"):
            limit_value = config.get(f"tool_max_attempts_{name}", None)

        # 3) 兜底全局默认
        if limit_value is None:
            limit_value = default_per_tool
        try:
            per_tool_limits[name] = max(1, int(limit_value))
        except (TypeError, ValueError):
            per_tool_limits[name] = default_per_tool

    return ToolCallBudget(
        per_tool_limits=per_tool_limits,
        default_per_tool=default_per_tool,
        total_budget=total_budget,
        invalid_limit=invalid_limit,
        relevance_check_call=relevance_check_call,
    )