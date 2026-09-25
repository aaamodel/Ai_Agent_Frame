# -*- coding: utf-8 -*-
"""状态图重构（2.1 统一 ToolCall + 2.2 StateGraph）最小自动化回归。

覆盖：
1. builder 四个条件路由纯函数；
2. FC / 文本两路 ToolCall 归一后走同一执行管线的对拍（2.1 防回归）；
3. Fake model_router + InMemorySaver 全链路：
   - react FC 多步正常收尾；
   - plan 空数据 → 换源 replan → 推理子任务 → 汇总；
   - 危险工具 interrupt 暂停 → resume 批准 / 拒绝。

直接运行：
    cd 项目根
    python -m pytest 测试/test_graph_state_machine.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())))

from langgraph.checkpoint.memory import InMemorySaver

from app.core.agent.graph.builder import (
    compile_agent_graph,
    route_after_execute,
    route_after_prepare,
    route_after_reflect,
    route_after_replan,
    route_after_summarize,
)
from app.core.agent.graph.deps import GraphDeps
from app.core.agent.graph.runner import GraphRunner
from app.core.agent.graph.checkpoint import aget_snapshot
from app.core.agent.graph.nodes._common import agent_goal_from_state
from app.core.agent.evidence import ingest_observation
from app.core.agent.evidence.view import plan_view_uids
from app.core.agent.orchestrator import IntentContext
from app.core.agent.react_agent import _parse_react_step
from app.core.agent.toolcall import (
    execute_tool_call,
    tool_call_from_text,
    tool_calls_from_fc,
)
from app.core.tools.registry import build_tool_call_budget
from app.core.agent.graph.state import REFLECT_RETRY_KEY


# ---------------------------------------------------------------------------
# Fake 基础设施
# ---------------------------------------------------------------------------
class FakeTool:
    def __init__(self, name: str, output: str = "正常观测结果") -> None:
        self.name = name
        self.description = f"{name} 测试工具"
        self.SYSTEM_PROMPT = ""
        self.parameters: List[Any] = []
        self._output = output

    def schema_parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"q": {"type": "string"}}}

    async def execute(self, **kwargs: Any) -> str:
        return f"{self.name} <- {json.dumps(kwargs, ensure_ascii=False)} => {self._output}"


class FakeRegistry:
    def __init__(self, tools: Dict[str, FakeTool]) -> None:
        self._tools = dict(tools)
        self.invocations: List[Dict[str, Any]] = []

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool(self, name: str) -> Optional[FakeTool]:
        return self._tools.get(name)

    async def invoke(self, name: str, arguments: Dict[str, Any]) -> str:
        self.invocations.append({"name": name, "arguments": arguments})
        tool = self._tools[name]
        return await tool.execute(**(arguments or {}))


class FakeMemory:
    def __init__(self) -> None:
        self.turns: List[Dict[str, Any]] = []

    async def get_context(self, session_id: str, query: str, limit: int = 6) -> Any:
        return SimpleNamespace(short_term_messages=[], long_term_items=[])

    async def append_turn(self, session_id: str, role: str, content: str, metadata: Any = None) -> None:
        self.turns.append({"session_id": session_id, "role": role, "content": content})


class FakeSkillManager:
    def __init__(self) -> None:
        self.state = SimpleNamespace(available_skills={})

    async def scan_and_refresh_skills(self) -> None:
        return None

    def resolve_relevant_skill(self, user_input: str) -> Dict[str, Any]:
        return {}


class FakeTracer:
    def new_trace_id(self) -> str:
        return "trace-test"

    def start_span(self, name: str, trace_id: str, attributes: Any = None) -> Any:
        return SimpleNamespace(name=name)

    def end_span(self, span: Any, error: Any = None) -> None:
        return None

    def log_event(self, trace_id: str, event: str, payload: Any = None) -> None:
        return None


class FakeModelRouter:
    """按调用形态脚本化返回。

    - chat（最终总结助手 system）：按 summary_verdicts 脚本依次返回结构化判定 JSON；
    - chat(response_format 非空)：第 1 次给取数计划（plan_tools），之后给纯推理
      replan 计划；
    - chat(普通)：按 system 提示词区分"子任务提炼"与（兜底）；
    - chat_with_tools(react)：消息中已含 role=tool 就给 Final Answer，否则给工具调用；
    - chat_with_tools(planner 强制取参)：从 messages 文本中识别当前子任务工具名回传。
    """

    DEFAULT_VERDICTS = [{
        "sufficient": True,
        "answer": "最终汇总：该数据源暂无记录，建议稍后再试。",
        "missing_info": "",
        "suggestion": "",
    }]

    def __init__(
        self,
        *,
        react_tool: str = "echo_tool",
        react_args: Optional[dict] = None,
        plan_tools: Optional[List[str]] = None,
        summary_verdicts: Optional[List[dict]] = None,
        subtask_outcomes: Optional[List[dict]] = None,
        plan_fc_args: Optional[Dict[str, dict]] = None,
        plan_fc_arg_sequences: Optional[Dict[str, List[dict]]] = None,
    ) -> None:
        """``subtask_outcomes``：脚本化「子任务提炼」的控制协议返回，
        用于验证跳过 / 提前收尾是否真的让后续子任务不再执行。

        ``plan_fc_args``：按工具名脚本化 planner FC 强制取参的返回
        （默认 {"q": "x"}），用于验证下游写工具用前序结论重组参数。

        ``plan_fc_arg_sequences``：按工具名给出 FC 返回序列，每次调用
        消费一个（耗尽后停在最后一个）。用于模拟真实 LLM 在 interrupt
        重放时的非确定性漂移——修复后重放根本不应再调 FC。"""
        self._react_tool = react_tool
        self._react_args = react_args or {"q": "北京天气"}
        self._plan_tools = list(plan_tools if plan_tools is not None else ["echo_tool"])
        self._summary_verdicts = list(summary_verdicts or self.DEFAULT_VERDICTS)
        self._plan_calls = 0
        self._summary_calls = 0
        self.summary_call_count = 0  # 对外只读：实际发生的 summarize 调用次数
        self._subtask_calls = 0
        self._subtask_outcomes = list(subtask_outcomes or [])
        self._plan_fc_args = dict(plan_fc_args or {})
        self._plan_fc_sequences = {
            name: list(seq) for name, seq in (plan_fc_arg_sequences or {}).items()
        }
        self._plan_fc_taken: Dict[str, int] = {}
        # 对外只读：planner 用途的 FC 取参被调了几次、分别给哪个工具
        self.fc_plan_calls: List[str] = []

    async def chat(self, messages: Any, *, purpose_hint: str = "", **kwargs: Any) -> Any:
        system_text = messages[0].get("content", "") if messages else ""

        if "最终总结助手" in system_text:
            idx = min(self._summary_calls, len(self._summary_verdicts) - 1)
            self._summary_calls += 1
            self.summary_call_count = self._summary_calls
            return SimpleNamespace(
                content=json.dumps(self._summary_verdicts[idx], ensure_ascii=False),
                reasoning_content="",
            )

        # plan 子任务提炼（控制协议结构化输出）
        # ⚠️ 必须排在 response_format 判断**之前**：提炼调用现在也带 response_format
        #（控制指令与结论在同一次调用产出，这是本变更的核心），因此"是否带
        # response_format" 已不能再用来区分规划与提炼，只能按 system 提示词判别。
        if "子任务执行专家" in system_text:
            default_outcome = {
                "conclusion": "子任务结论：数据源无记录，无法提供相关数字。",
                "solved": "yes", "next_action": "continue",
                "skip_task_ids": None, "reason": "",
            }
            if self._subtask_outcomes:
                idx = min(self._subtask_calls, len(self._subtask_outcomes) - 1)
                default_outcome = self._subtask_outcomes[idx]
            self._subtask_calls += 1
            return SimpleNamespace(
                content=json.dumps(default_outcome, ensure_ascii=False), reasoning_content=""
            )

        if kwargs.get("response_format") is not None:
            self._plan_calls += 1
            if self._plan_calls == 1 and self._plan_tools:
                subtasks = [{
                    "id": f"t{i + 1}",
                    "title": f"取数{i + 1}",
                    "description": f"调用 {name} 取数",
                    "action_type": "tool",
                    "tool_name": name,
                    "tool_args_hint": json.dumps({"q": "x"}, ensure_ascii=False),
                } for i, name in enumerate(self._plan_tools)]
            else:
                subtasks = [{
                    "id": "r1", "title": "基于已有信息推理",
                    "description": "无可用外部数据，直接推理收尾",
                    "action_type": "reasoning",
                }]
            content = json.dumps({"subtasks": subtasks}, ensure_ascii=False)
            return SimpleNamespace(content=content, reasoning_content="")

        raise AssertionError(f"FakeModelRouter.chat 收到未预期的调用: {system_text[:80]}")

    async def chat_with_tools(
        self, messages: Any, tools: Any, tool_choice: Any = None, *,
        purpose_hint: str = "", **kwargs: Any,
    ) -> Any:
        if purpose_hint == "planner":
            # 从取参请求文本中识别当前子任务声明的工具名
            blob = json.dumps(messages, ensure_ascii=False, default=str)
            candidates = self._plan_tools + [self._react_tool]
            chosen = next((n for n in candidates if n and n in blob),
                          self._plan_tools[0] if self._plan_tools else self._react_tool)
            self.fc_plan_calls.append(chosen)
            sequence = self._plan_fc_sequences.get(chosen)
            if sequence:
                idx = min(self._plan_fc_taken.get(chosen, 0), len(sequence) - 1)
                self._plan_fc_taken[chosen] = idx + 1
                fc_arguments = sequence[idx]
            else:
                fc_arguments = self._plan_fc_args.get(chosen, {"q": "x"})
            return SimpleNamespace(
                content="", reasoning_content="",
                tool_calls=[{"id": "fc-plan-1", "function": {
                    "name": chosen,
                    "arguments": json.dumps(fc_arguments, ensure_ascii=False),
                }}],
            )

        has_tool_message = any((m or {}).get("role") == "tool" for m in messages)
        if has_tool_message:
            return SimpleNamespace(content="Final Answer: 已根据工具结果作答。", reasoning_content="", tool_calls=[])
        return SimpleNamespace(
            content="", reasoning_content="",
            tool_calls=[{"id": "fc-react-1", "function": {
                "name": self._react_tool,
                "arguments": json.dumps(self._react_args, ensure_ascii=False),
            }}],
        )


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _base_config(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "max_replan_attempts": 1,
        "react_max_steps": 5,
        "enable_skill_tool_gating": False,
        "enable_empty_result_replan": True,
    }
    cfg.update(overrides)
    return cfg


def _build_runner(tools: Dict[str, FakeTool], config: Dict[str, Any],
                  model_router: Optional[FakeModelRouter] = None) -> tuple:
    saver = InMemorySaver()
    graph = compile_agent_graph(saver)
    runner = GraphRunner(graph, saver)
    registry = FakeRegistry(tools)
    deps = GraphDeps(
        config=config,
        model_router=model_router or FakeModelRouter(),
        memory=FakeMemory(),
        tools=registry,
        skill_manager=FakeSkillManager(),
        tracer=FakeTracer(),
    )
    return runner, deps, registry, deps.memory


# ---------------------------------------------------------------------------
# 1. 路由纯函数
# ---------------------------------------------------------------------------
def test_route_after_prepare() -> None:
    assert route_after_prepare({"should_plan": True}) == "plan"
    assert route_after_prepare({"should_plan": False}) == "execute"


def test_route_after_execute() -> None:
    # 终局答案 → summarize
    assert route_after_execute({"final_answer": "done", "react_step": 0}) == "summarize"

    # plan 未跑完：步级空/错信号不再中断计划，继续下一子任务
    assert route_after_execute({
        "final_answer": "",
        "plan": [{"id": "a"}, {"id": "b"}], "cursor": 0,
        "empty_data_signal": "sig", "last_error": "err",
        "replan_attempts": 0, "max_replan": 2,
    }) == "execute"

    # plan 跑完 + L2/L3 证据不足信号 + **方向性错误** + 有次数与预算余量 → replan
    assert route_after_execute({
        "final_answer": "", "plan": [{"id": "a"}], "cursor": 1,
        "insufficiency_signal": "sig", "insufficiency_kind": "off_topic",
        "replan_attempts": 0, "max_replan": 2,
        "budget": {"total_budget": 20, "total_used": 1},
    }) == "replan"

    # 证据不足但缺口性质是"没取到数据"（非方向性错误）→ 不再整轮重规划
    assert route_after_execute({
        "final_answer": "", "plan": [{"id": "a"}], "cursor": 1,
        "insufficiency_signal": "sig", "insufficiency_kind": "no_data",
        "replan_attempts": 0, "max_replan": 2,
        "budget": {"total_budget": 20, "total_used": 1},
    }) == "summarize"

    # 次数耗尽 → summarize（不再 replan）
    assert route_after_execute({
        "final_answer": "", "plan": [{"id": "a"}], "cursor": 1,
        "insufficiency_signal": "sig",
        "replan_attempts": 2, "max_replan": 2,
        "budget": {"total_budget": 20, "total_used": 1},
    }) == "summarize"

    # 总预算耗尽 → summarize
    assert route_after_execute({
        "final_answer": "", "plan": [{"id": "a"}], "cursor": 1,
        "insufficiency_signal": "sig",
        "replan_attempts": 0, "max_replan": 2,
        "budget": {"total_budget": 20, "total_used": 20},
    }) == "summarize"

    # total_budget=0 表示不限总量 → 仍可 replan
    assert route_after_execute({
        "final_answer": "", "plan": [{"id": "a"}], "cursor": 1,
        "insufficiency_signal": "sig", "insufficiency_kind": "off_topic",
        "replan_attempts": 0, "max_replan": 2,
        "budget": {"total_budget": 0, "total_used": 99},
    }) == "replan"

    # 计划跑完但无证据不足信号 → 正常 summarize
    assert route_after_execute({"final_answer": "", "plan": [{"id": "a"}], "cursor": 1}) == "summarize"

    # 旧步级信号单独存在（无 insufficiency_signal、无计划）不再驱动 replan：
    # 落到 react 步数判定 / 兜底 summarize
    assert route_after_execute({
        "final_answer": "", "plan": [], "empty_data_signal": "sig", "last_error": "err",
        "replan_attempts": 0, "max_replan": 2,
        "react_step": 5, "max_steps": 5,
    }) == "summarize"

    # react 未达上限 → 自环；达上限 → summarize
    assert route_after_execute({"final_answer": "", "react_step": 2, "max_steps": 5}) == "execute"
    assert route_after_execute({"final_answer": "", "react_step": 5, "max_steps": 5}) == "summarize"


def test_route_after_summarize() -> None:
    # 无信号 → persist
    assert route_after_summarize({}) == "persist"
    # L3 证据不足且**方向性错误**且有余量 → replan
    assert route_after_summarize({
        "insufficiency_signal": "sig", "insufficiency_kind": "off_topic",
        "replan_attempts": 0, "max_replan": 2,
        "budget": {"total_budget": 20, "total_used": 3},
    }) == "replan"
    # 证据不足但只是"没取到数据" → persist（步级问题不该付整轮代价）
    assert route_after_summarize({
        "insufficiency_signal": "sig", "insufficiency_kind": "no_data",
        "replan_attempts": 0, "max_replan": 2,
        "budget": {"total_budget": 20, "total_used": 3},
    }) == "persist"
    # 次数/预算耗尽 → persist（节点已用草稿填好 final_answer）
    assert route_after_summarize({
        "insufficiency_signal": "sig",
        "replan_attempts": 2, "max_replan": 2,
    }) == "persist"
    assert route_after_summarize({
        "insufficiency_signal": "sig",
        "replan_attempts": 0, "max_replan": 2,
        "budget": {"total_budget": 10, "total_used": 10},
    }) == "persist"


def test_route_after_replan() -> None:
    assert route_after_replan({"plan": [{"id": "a"}]}) == "execute"
    assert route_after_replan({"plan": []}) == "summarize"
    assert route_after_replan({"plan": [], "last_error": "x"}) == "summarize"


def test_route_after_reflect() -> None:
    assert route_after_reflect({"reflect_failed": False}) == "summarize"
    state = {"reflect_failed": True, "retry_counts": {REFLECT_RETRY_KEY: 1}}
    assert route_after_reflect(state, None) == "execute"
    state = {"reflect_failed": True, "retry_counts": {REFLECT_RETRY_KEY: 5}}
    assert route_after_reflect(state, None) == "summarize"


# ---------------------------------------------------------------------------
# 2. 2.1：FC / 文本两路 ToolCall 归一对拍
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fc_and_text_share_same_pipeline() -> None:
    registry = FakeRegistry({"echo_tool": FakeTool("echo_tool", "两路观测一致")})
    budget = build_tool_call_budget({"tool_max_attempts": 10}, registry, ["echo_tool"])

    fc_call = tool_calls_from_fc([{"id": "c1", "function": {
        "name": "echo_tool", "arguments": json.dumps({"q": "x"}, ensure_ascii=False),
    }}], 0)[0]
    parsed = _parse_react_step(
        'Thought: 需要取数\nAction: echo_tool\nAction Input: {"q": "x"}'
    )
    text_call = tool_call_from_text(parsed)
    assert text_call is not None

    r1 = await execute_tool_call(fc_call, registry, call_budget=budget)
    r2 = await execute_tool_call(text_call, registry, call_budget=budget)

    assert r1.status == r2.status == "ok"
    assert r1.invoked and r2.invoked
    assert r1.observation == r2.observation
    assert budget._used_per_tool["echo_tool"] == 2
    assert fc_call.source == "fc" and text_call.source == "text"


# ---------------------------------------------------------------------------
# 3a. 全链路：react FC 多步
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_react_fc_happy_path() -> None:
    runner, deps, registry, memory = _build_runner(
        {"echo_tool": FakeTool("echo_tool", "北京今天晴，25℃")}, _base_config()
    )
    outcome = await runner.run(
        deps=deps, user_input="北京天气如何", session_id="s-react", mode="react",
        intent=IntentContext(),
    )

    assert outcome.paused is False
    assert outcome.response.success is True
    assert "已根据工具结果作答" in outcome.response.answer
    assert outcome.response.mode_used == "react"
    assert len(registry.invocations) == 1
    assert registry.invocations[0]["name"] == "echo_tool"
    # persist 双写短期记忆
    assert [t["role"] for t in memory.turns] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# 3b. 全链路：plan 唯一工具空数据 → L2 全坏闸门 → replan 换推理 → 汇总（degraded=True）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plan_empty_data_replan() -> None:
    runner, deps, registry, memory = _build_runner(
        {"echo_tool": FakeTool("echo_tool", "未找到相关数据")},
        _base_config(max_replan_attempts=1),
    )
    outcome = await runner.run(
        deps=deps, user_input="查某指标", session_id="s-plan", mode="plan_execute",
        intent=IntentContext(),
    )

    assert outcome.paused is False
    assert outcome.response.success is True
    assert outcome.response.mode_used == "plan_execute"
    # ⚠️ 触发收窄后不再 replan，因此不再被标记为 degraded——该标记目前由
    # replan_node 写入。本次仍以"暂无记录"诚实收尾，只是不再付出整轮代价。
    assert outcome.response.degraded is False
    assert "暂无记录" in outcome.response.answer
    # 取数工具只调用一次（空数据记账后计划收尾）
    assert len(registry.invocations) == 1
    # 触发收窄后：缺口只是"没取到数据"（非方向性错误）→ **不再整轮重规划**，
    # 因此只有初始 1 次规划；summarize 1 次即以诚实的部分答案收尾。
    assert deps.model_router._plan_calls == 1
    assert deps.model_router.summary_call_count == 1
    assert [t["role"] for t in memory.turns] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# 3b-2. 全链路：两个工具一空一成 → 计划跑完，不 replan，直接 summarize
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plan_partial_bad_continues_without_replan() -> None:
    router = FakeModelRouter(plan_tools=["empty_tool", "echo_tool"])
    runner, deps, registry, _ = _build_runner(
        {
            "empty_tool": FakeTool("empty_tool", "未找到相关数据"),
            "echo_tool": FakeTool("echo_tool", "查到退货率为 12%，环比下降 2pct"),
        },
        _base_config(max_replan_attempts=2),
        model_router=router,
    )
    outcome = await runner.run(
        deps=deps, user_input="查退货率", session_id="s-partial", mode="plan_execute",
        intent=IntentContext(),
    )

    assert outcome.response.success is True
    assert outcome.response.degraded is False
    # 两个工具各调用一次（空数据步不中断，第二个继续跑）
    assert [inv["name"] for inv in registry.invocations] == ["empty_tool", "echo_tool"]
    # 没有 replan：只有初始 1 次规划 LLM；summarize 1 次判定 sufficient
    assert router._plan_calls == 1
    assert router.summary_call_count == 1


# ---------------------------------------------------------------------------
# 3b-3. 全链路：工具返回跑题内容（ok 但不相关）→ L3 summarize 判不足 →
#        带缺口 replan → 二次汇总 sufficient
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plan_irrelevant_result_summary_triggers_replan() -> None:
    router = FakeModelRouter(
        plan_tools=["echo_tool"],
        summary_verdicts=[
            # 跑题＝方向性错误，只有这种缺口才允许重规划
            {"sufficient": False, "answer": "草稿：只有无关信息",
             "missing_info": "缺少退货率数据", "suggestion": "改用售后类数据源",
             "gap_kind": "off_topic"},
            {"sufficient": True, "answer": "补数后最终答案：退货率为 12%。",
             "missing_info": "", "suggestion": ""},
        ],
    )
    runner, deps, registry, _ = _build_runner(
        {"echo_tool": FakeTool("echo_tool", "今日体育新闻：某足球比赛 2:1 结束")},
        _base_config(max_replan_attempts=1),
        model_router=router,
    )
    outcome = await runner.run(
        deps=deps, user_input="查退货率", session_id="s-irrelevant", mode="plan_execute",
        intent=IntentContext(),
    )

    assert outcome.response.success is True
    assert outcome.response.degraded is True
    assert outcome.response.answer == "补数后最终答案：退货率为 12%。"
    # 原工具只调 1 次（replan fallback 是纯推理，无新工具）
    assert len(registry.invocations) == 1
    # 初始 plan + 1 次 replan；summarize 共 2 次（不足 → 充分）
    assert router._plan_calls == 2
    assert router.summary_call_count == 2


# ---------------------------------------------------------------------------
# 3b-4. 全链路：L3 判不足但 replan 额度=0 → 直接用草稿降级收尾，不发起 replan
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plan_insufficient_exhausted_uses_draft() -> None:
    router = FakeModelRouter(
        plan_tools=["echo_tool"],
        summary_verdicts=[
            {"sufficient": False, "answer": "部分答案：仅知出货量，退货率未知",
             "missing_info": "退货率", "suggestion": "售后系统"},
        ],
    )
    runner, deps, registry, _ = _build_runner(
        {"echo_tool": FakeTool("echo_tool", "出货量 1000 件")},
        _base_config(max_replan_attempts=0),
        model_router=router,
    )
    outcome = await runner.run(
        deps=deps, user_input="查退货率", session_id="s-draft", mode="plan_execute",
        intent=IntentContext(),
    )

    assert outcome.response.degraded is True
    assert outcome.response.success is True  # 有草稿可用
    assert outcome.response.answer == "部分答案：仅知出货量，退货率未知"
    assert router._plan_calls == 1  # 无 replan LLM
    assert router.summary_call_count == 1
    assert len(registry.invocations) == 1


# ---------------------------------------------------------------------------
# 3c. 全链路：危险工具 interrupt → resume 批准 / 拒绝
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_danger_tool_pause_and_approve() -> None:
    cfg = _base_config(agent_approval_enabled=True, agent_danger_tools="danger_tool")
    runner, deps, registry, _ = _build_runner(
        {"danger_tool": FakeTool("danger_tool", "危险操作已执行，结果OK")},
        cfg, model_router=FakeModelRouter(react_tool="danger_tool", react_args={"target": "x"}),
    )
    outcome = await runner.run(
        deps=deps, user_input="执行危险操作", session_id="s-approve", mode="react",
        intent=IntentContext(),
    )

    # 首次运行：挂起等审批，工具未执行
    assert outcome.paused is True
    assert outcome.response.awaiting_approval is True
    assert len(outcome.approval_payloads) == 1
    payload = outcome.approval_payloads[0]
    assert payload["tool_name"] == "danger_tool"
    assert payload["run_id"] == outcome.run_id
    assert registry.invocations == []

    # resume：批准（重建 deps，模拟新 HTTP 请求）
    resume_deps = GraphDeps(
        config=cfg, model_router=deps.model_router, memory=FakeMemory(),
        tools=registry, skill_manager=FakeSkillManager(), tracer=FakeTracer(),
    )
    resumed = await runner.resume(run_id=outcome.run_id, deps=resume_deps, approved=True)
    assert resumed.paused is False
    assert resumed.response.success is True
    assert "已根据工具结果作答" in resumed.response.answer
    assert len(registry.invocations) == 1


@pytest.mark.asyncio
async def test_danger_tool_pause_and_deny() -> None:
    cfg = _base_config(agent_approval_enabled=True, agent_danger_tools="danger_tool")
    runner, deps, registry, _ = _build_runner(
        {"danger_tool": FakeTool("danger_tool", "不应出现的结果")},
        cfg, model_router=FakeModelRouter(react_tool="danger_tool", react_args={"target": "x"}),
    )
    outcome = await runner.run(
        deps=deps, user_input="执行危险操作", session_id="s-deny", mode="react",
        intent=IntentContext(),
    )
    assert outcome.paused is True

    resume_deps = GraphDeps(
        config=cfg, model_router=deps.model_router, memory=FakeMemory(),
        tools=registry, skill_manager=FakeSkillManager(), tracer=FakeTracer(),
    )
    resumed = await runner.resume(
        run_id=outcome.run_id, deps=resume_deps, approved=False, comment="禁止",
    )
    assert resumed.paused is False
    assert resumed.response.success is True
    # 拒绝后工具绝不执行，模型走正常推理收尾
    assert registry.invocations == []
    assert "已根据工具结果作答" in resumed.response.answer


# ---------------------------------------------------------------------------
# 3c-2. plan_execute 路径：危险工具 interrupt 同样必须真正暂停图
#   历史事故（2026-09-23，langgraph 1.2.11）：GraphInterrupt 继承 Exception，
#   _execute_plan_step 的宽 except Exception 把审批中断当成步级错误吞掉，
#   图没暂停、继续跑完并友好降级，前端永远收不到 awaiting_approval。
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plan_danger_tool_pause_and_approve() -> None:
    cfg = _base_config(agent_approval_enabled=True, agent_danger_tools="danger_tool")
    runner, deps, registry, _ = _build_runner(
        {"danger_tool": FakeTool("danger_tool", "危险报告已导出，结果OK")},
        cfg, model_router=FakeModelRouter(plan_tools=["danger_tool"]),
    )
    outcome = await runner.run(
        deps=deps, user_input="查数并导出危险报告", session_id="s-plan-approve",
        mode="plan_execute", intent=IntentContext(),
    )

    # 首次运行：必须真正挂起等审批，工具一次都没执行
    assert outcome.paused is True
    assert outcome.response.awaiting_approval is True
    assert len(outcome.approval_payloads) == 1
    payload = outcome.approval_payloads[0]
    assert payload["tool_name"] == "danger_tool"
    assert payload["subtask_id"] == "t1"
    assert registry.invocations == []

    resume_deps = GraphDeps(
        config=cfg, model_router=deps.model_router, memory=FakeMemory(),
        tools=registry, skill_manager=FakeSkillManager(), tracer=FakeTracer(),
    )
    resumed = await runner.resume(run_id=outcome.run_id, deps=resume_deps, approved=True)
    assert resumed.paused is False
    assert resumed.response.success is True
    # 批准后节点重放、闸门放行：工具恰好执行 1 次（不能因重放重复执行）
    assert len(registry.invocations) == 1


@pytest.mark.asyncio
async def test_plan_danger_tool_pause_and_deny() -> None:
    cfg = _base_config(agent_approval_enabled=True, agent_danger_tools="danger_tool")
    runner, deps, registry, _ = _build_runner(
        {"danger_tool": FakeTool("danger_tool", "不应出现的结果")},
        cfg, model_router=FakeModelRouter(plan_tools=["danger_tool"]),
    )
    outcome = await runner.run(
        deps=deps, user_input="查数并导出危险报告", session_id="s-plan-deny",
        mode="plan_execute", intent=IntentContext(),
    )
    assert outcome.paused is True

    resume_deps = GraphDeps(
        config=cfg, model_router=deps.model_router, memory=FakeMemory(),
        tools=registry, skill_manager=FakeSkillManager(), tracer=FakeTracer(),
    )
    resumed = await runner.resume(
        run_id=outcome.run_id, deps=resume_deps, approved=False, comment="禁止",
    )
    assert resumed.paused is False
    # 拒绝：工具零执行、不触发再次暂停、不崩溃；plan 路径会诚实告知导出未完成
    # （react 路径则由模型把拒绝观测推理成 Final Answer，两者收尾形态不同）
    assert registry.invocations == []
    assert resumed.response.awaiting_approval is False
    assert resumed.response.degraded is True


# ---------------------------------------------------------------------------
# 3c-3. plan_execute 下游写工具：必须用 FC 结合前序真实结论重组参数
#   历史事故（2026-09-23）：planner 看不到运行结果，给导出工具的正文只写了
#   "××排名数据"标题性占位；hint 必填齐全 → 零 LLM 直达工具 → 空报表。
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plan_downstream_danger_tool_composes_args_from_prior_results() -> None:
    cfg = _base_config(agent_approval_enabled=True, agent_danger_tools="danger_tool")
    router = FakeModelRouter(
        plan_tools=["echo_tool", "danger_tool"],
        plan_fc_args={"danger_tool": {"q": "基于前序结论组织的真实正文：张伟4单"}},
    )
    runner, deps, registry, _ = _build_runner(
        {"echo_tool": FakeTool("echo_tool", "排名：张伟4单、王强4单"),
         "danger_tool": FakeTool("danger_tool", "报表已导出")},
        cfg, model_router=router,
    )
    outcome = await runner.run(
        deps=deps, user_input="查数并导出报告", session_id="s-compose",
        mode="plan_execute", intent=IntentContext(),
    )

    # 首次运行在 danger_tool 审批闸门暂停
    assert outcome.paused is True
    # 第一个只读工具零 FC 直用 hint；下游写工具即便 hint 齐全也强制 FC 组参
    assert router.fc_plan_calls == ["danger_tool"]
    danger_calls_before = [i for i in registry.invocations if i["name"] == "danger_tool"]
    assert danger_calls_before == []  # 未批准前不执行

    resume_deps = GraphDeps(
        config=cfg, model_router=router, memory=FakeMemory(),
        tools=registry, skill_manager=FakeSkillManager(), tracer=FakeTracer(),
    )
    resumed = await runner.resume(run_id=outcome.run_id, deps=resume_deps, approved=True)
    assert resumed.paused is False

    danger_calls = [i for i in registry.invocations if i["name"] == "danger_tool"]
    assert len(danger_calls) == 1  # 节点重放不重复执行
    # 实际入参必须是 FC 基于前序结论生成的值，而不是 planner 的占位 hint
    assert danger_calls[0]["arguments"]["q"] == "基于前序结论组织的真实正文：张伟4单"


@pytest.mark.asyncio
async def test_plan_danger_tool_approved_arguments_stable_across_replay() -> None:
    """审批卡片上看到的字段必须与批准后实际执行的字段逐字一致。

    历史事故（2026-09-23 实测）：interrupt 恢复时整个 execute 节点从头
    重放，plan 路径再次调用非确定性 FC 参数填充 LLM——首次生成正文 A
    （审批卡片显示 A），重放漂移成正文 B，工具最终带着 B 落盘。
    修复后：重放必须复用审批载荷中的入参，且不再发生第二次 FC 调用。
    """
    cfg = _base_config(agent_approval_enabled=True, agent_danger_tools="danger_tool")
    body_a = "结论：张伟4单（审批卡片上看到的正文）"
    body_b = "结论：张伟4单（重放漂移出来的另一段正文，绝不能被执行）"
    router = FakeModelRouter(
        plan_tools=["echo_tool", "danger_tool"],
        plan_fc_arg_sequences={"danger_tool": [{"q": body_a}, {"q": body_b}]},
    )
    runner, deps, registry, _ = _build_runner(
        {"echo_tool": FakeTool("echo_tool", "排名：张伟4单"),
         "danger_tool": FakeTool("danger_tool", "报表已导出")},
        cfg, model_router=router,
    )
    outcome = await runner.run(
        deps=deps, user_input="查数并导出报告", session_id="s-approve-replay",
        mode="plan_execute", intent=IntentContext(),
    )
    assert outcome.paused is True
    payload = outcome.approval_payloads[0]
    assert payload["arguments"]["q"] == body_a
    assert router.fc_plan_calls == ["danger_tool"]  # 暂停前仅 1 次 FC

    resume_deps = GraphDeps(
        config=cfg, model_router=router, memory=FakeMemory(),
        tools=registry, skill_manager=FakeSkillManager(), tracer=FakeTracer(),
    )
    resumed = await runner.resume(run_id=outcome.run_id, deps=resume_deps, approved=True)
    assert resumed.paused is False
    assert resumed.response.success is True

    # 重放不得再次调用 FC（非确定性漂移源被物理消除）
    assert router.fc_plan_calls == ["danger_tool"]
    danger_calls = [i for i in registry.invocations if i["name"] == "danger_tool"]
    assert len(danger_calls) == 1
    # 实际执行入参 = 审批时看到的正文 A，不能是漂移的 B
    assert danger_calls[0]["arguments"]["q"] == body_a


# ---------------------------------------------------------------------------
# 3d. 控制协议：跳过子任务 / 提前收尾 —— 后续子任务必须**不再执行**
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_control_skip_task_not_executed() -> None:
    """模型声明跳过 t2 → t2 的工具绝不执行，计划仍正常收尾。"""
    router = FakeModelRouter(
        plan_tools=["echo_tool", "echo_tool"],
        subtask_outcomes=[{
            "conclusion": "已拿到退货率 12%", "solved": "yes",
            "next_action": "continue", "skip_task_ids": ["t2"],
            "reason": "t1 已取得所需答案",
        }],
    )
    runner, deps, registry, _ = _build_runner(
        {"echo_tool": FakeTool("echo_tool", "查到退货率为 12%")},
        _base_config(max_replan_attempts=2),
        model_router=router,
    )
    outcome = await runner.run(
        deps=deps, user_input="查退货率", session_id="s-skip", mode="plan_execute",
        intent=IntentContext(),
    )
    assert outcome.response.success is True
    # 计划有 2 个取数子任务，t2 被跳过 → 工具只调用 1 次
    assert len(registry.invocations) == 1
    assert router._plan_calls == 1  # 未触发 replan


@pytest.mark.asyncio
async def test_control_finish_skips_remaining() -> None:
    """模型判定答案已足够 → 剩余子任务全部不执行，直接进汇总。"""
    router = FakeModelRouter(
        plan_tools=["echo_tool", "echo_tool", "echo_tool"],
        subtask_outcomes=[{
            "conclusion": "已拿到退货率 12%", "solved": "yes",
            "next_action": "finish", "skip_task_ids": None, "reason": "已足以回答",
        }],
    )
    runner, deps, registry, _ = _build_runner(
        {"echo_tool": FakeTool("echo_tool", "查到退货率为 12%")},
        _base_config(max_replan_attempts=2),
        model_router=router,
    )
    outcome = await runner.run(
        deps=deps, user_input="查退货率", session_id="s-finish", mode="plan_execute",
        intent=IntentContext(),
    )
    assert outcome.response.success is True
    # 3 个取数子任务，第 1 个之后即提前收尾 → 只调用 1 次
    assert len(registry.invocations) == 1
    assert router.summary_call_count == 1


# ---------------------------------------------------------------------------
# 6.3 方向性错误场景：重规划恰好 1 次，之后不再触发（硬上限）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_off_topic_replans_exactly_once_then_stops() -> None:
    """全部结论跑题（方向性错误）→ 允许 1 次重规划；再次判定不足时诚实收尾。

    ⚠️ 即便配置把 max_replan_attempts 放到 2，硬上限仍是 1——"至多一次"是无条件
    约束，配置只能把它调得更小（`agent/replan-context` 的硬性上限需求）。
    """
    router = FakeModelRouter(
        plan_tools=["echo_tool"],
        summary_verdicts=[
            {"sufficient": False, "answer": "草稿：只有无关信息",
             "missing_info": "缺少退货率数据", "suggestion": "换数据源",
             "gap_kind": "off_topic"},
            {"sufficient": False, "answer": "仍是草稿：依然无关",
             "missing_info": "仍然缺少", "suggestion": "再换一个",
             "gap_kind": "off_topic"},
        ],
    )
    runner, deps, registry, _ = _build_runner(
        {"echo_tool": FakeTool("echo_tool", "今日体育新闻：某足球比赛 2:1 结束")},
        _base_config(max_replan_attempts=2),
        model_router=router,
    )
    outcome = await runner.run(
        deps=deps, user_input="查退货率", session_id="s-one-replan",
        mode="plan_execute", intent=IntentContext(),
    )

    assert outcome.paused is False
    # 初始 plan + 恰好 1 次 replan
    assert router._plan_calls == 2
    # 汇总两次判不足：第一次触发重规划，第二次不再触发
    assert router.summary_call_count == 2


# ---------------------------------------------------------------------------
# 8.3 端到端：取到了数据但数据不对 → 证据缺口 → 注入候选 → 模型选中后换源
# ---------------------------------------------------------------------------
_ASSET_TABLE = (
    "| 资产 | 位置 | 用途(调用工具） |\n"
    "|---|---|---|\n"
    "| 客户线索台账.xlsx | `raw_data/sales_intel/客户线索台账.xlsx` | 线索查询（alt_tool） |\n"
)
_QUESTION = "我们优先做哪些行业？哪些行业算次优先？"


@pytest.mark.asyncio
async def test_evidence_gap_lets_model_switch_data_source_midflight() -> None:
    """本 trace 的核心失败场景，端到端跑通。

    t1 返回了内容但与问题无关（体育新闻 vs 行业优先级）——**它不是空的、也不是
    错的**，所以"取数失败"类触发条件覆盖不到它。证据缺口检测必须把它抓住，
    注入候选方向列表，并让模型从封闭列表里选中一个替代方向。
    """
    router = FakeModelRouter(
        plan_tools=["t1_tool", "t2_tool"],
        subtask_outcomes=[
            {"conclusion": "本步返回与问题无关", "solved": "no",
             "next_action": "continue", "selected_alternative_id": "tool:alt_tool"},
            # 纠偏插入的 alt 步与顺延的 t2 步：中性产出，不再选择替代方向
            {"conclusion": "替代方向已取到行业优先级数据", "solved": "yes",
             "next_action": "continue", "selected_alternative_id": None},
        ],
        summary_verdicts=[
            {"sufficient": True, "answer": "已完成", "missing_info": "", "suggestion": ""}
        ],
    )
    runner, deps, registry, _ = _build_runner(
        {
            "t1_tool": FakeTool("t1_tool", _ASSET_TABLE + "今日体育新闻：某足球比赛 2:1 结束"),
            "t2_tool": FakeTool("t2_tool", "原始第二步结果"),
            "alt_tool": FakeTool("alt_tool", "替代方向取到了行业优先级数据"),
        },
        _base_config(),
        model_router=router,
    )

    outcome = await runner.run(
        deps=deps, user_input=_QUESTION, session_id="s-step-corr",
        mode="plan_execute", intent=IntentContext(),
    )

    assert outcome.paused is False
    names = [inv["name"] for inv in registry.invocations]
    assert names[0] == "t1_tool"
    # 纠偏在 t1 之后**插入** alt_tool 步：下一步立即执行；原 t2 不被替换，顺延执行
    assert names[1] == "alt_tool", f"就地纠偏未生效，实际调用序列={names}"
    assert names[2] == "t2_tool", f"原计划子任务应顺延保留，实际调用序列={names}"


@pytest.mark.asyncio
async def test_no_injection_when_step_is_healthy_so_plan_is_untouched() -> None:
    """对照实验：不满足注入条件时，模型给的标识必须被忽略、计划原样不动。

    这同时验证了"标识不在本次候选列表中 → 忽略该指令并降级为继续执行"——
    没有注入就没有候选集，因此任何标识都无效。
    """
    router = FakeModelRouter(
        plan_tools=["t1_tool", "t2_tool"],
        subtask_outcomes=[
            {"conclusion": "行业优先级已给出", "solved": "yes",
             "next_action": "continue", "selected_alternative_id": "tool:alt_tool"},
            # t2 步：即使证据缺口置位、候选可见，模型不选 → 计划不得被延长
            {"conclusion": "第二步完成", "solved": "yes",
             "next_action": "continue", "selected_alternative_id": None},
        ],
        summary_verdicts=[
            {"sufficient": True, "answer": "已完成", "missing_info": "", "suggestion": ""}
        ],
    )
    runner, deps, registry, _ = _build_runner(
        {
            "t1_tool": FakeTool("t1_tool", "行业优先级排序结论：金融行业优先，其次先进制造。"),
            "t2_tool": FakeTool("t2_tool", "第二步结果"),
            "alt_tool": FakeTool("alt_tool", "替代结果"),
        },
        _base_config(),
        model_router=router,
    )

    outcome = await runner.run(
        deps=deps, user_input=_QUESTION, session_id="s-step-corr-ctrl",
        mode="plan_execute", intent=IntentContext(),
    )

    assert outcome.paused is False
    assert [inv["name"] for inv in registry.invocations] == ["t1_tool", "t2_tool"]


# ---------------------------------------------------------------------------
# 9. 证据板 T5（D10）：plan 延迟一步证据回取端到端
# ---------------------------------------------------------------------------
_RESTORE_BODIES = [
    "退款审批须上传签收凭证与发票照片，财务在材料齐全后三个工作日内完成审核，"
    "审核通过的款项原路退回付款账户，遇法定节假日顺延，跨月提交的单据并入下一结算周期统一处理。",
    "退货商品入库验收由仓储岗负责，外包装破损或附件缺失的包裹需现场拍照登记，"
    "验收不通过的退货单退回客服跟进，客户补充材料后重新发起流程，验收通过才释放退款额度。",
    "运费险理赔在退款完成后自动触发，理赔金额按收货与退货两段实际运费计算，"
    "三个工作日内发放至客户下单时使用的支付账户，客户可在订单详情页查看理赔进度与到账记录。",
    "大额退款（单笔超过一千元）须财务主管二次复核，复核内容包括订单真实性与发票状态，"
    "每月五日与二十日为大额退款集中打款日，紧急情形可申请单独走款但需分管总监邮件审批。",
    "优惠券与积分抵扣部分按原渠道分别退回：平台券退回卡券包且有效期不延长，"
    "积分退回会员账户并恢复成长值，第三方支付的差额部分按原路退回，组合支付订单逐笔算清。",
    "跨境订单退款涉及汇率波动，按下单时锁定的结算汇率折算外币，"
    "关税与清关服务费不在退款范围内，银行端国际汇款一般需要五到七个工作日，到账短信可能延迟。",
    "质量问题导致的退货运费由商家承担，客户先行垫付后凭快递底单报销，"
    "七天无理由退货的往返运费由客户自行承担，拒收包裹产生的退回运费同样从退款金额中扣减。",
    "退款纠纷统一由售后专员建单跟进，协商记录全程留痕，"
    "超过十五天未达成一致的工单升级至平台介入，平台依据聊天记录与物流凭证在七个工作日内作出裁决。",
]


def _restore_rag_obs() -> str:
    body = "".join(
        f"[{i + 1}] 来源文献: policy_{i + 1}.txt\n内容片段: {text}\n"
        for i, text in enumerate(_RESTORE_BODIES)
    )
    return f"--- 知识库检索结果 (查询: 退款政策) ---\n{body}"


class _RestoreModelRouter(FakeModelRouter):
    """记录每次子任务提炼的入参消息，供断言恢复步确实再提炼了一次。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.subtask_distill_messages: List[Any] = []

    async def chat(self, messages: Any, *, purpose_hint: str = "", **kwargs: Any) -> Any:
        system_text = messages[0].get("content", "") if messages else ""
        if "子任务执行专家" in system_text:
            self.subtask_distill_messages.append(messages)
        return await super().chat(messages, purpose_hint=purpose_hint, **kwargs)


class _RecordingTracer(FakeTracer):
    def __init__(self) -> None:
        self.events: List[tuple] = []

    def log_event(self, trace_id: str, event: str, payload: Any = None) -> None:
        self.events.append((event, payload or {}))


@pytest.mark.asyncio
async def test_plan_evidence_restore_inserts_internal_step_and_redistills() -> None:
    """t1 提炼时索取 2 个被省略证据编号 → 插入零外部调用的内部恢复步 →
    回填原文 → 再走一次提炼；恢复步再次请求必须被忽略（每子任务一次）。"""
    rag_name = "rag_knowledge_search"
    inner_obs = _restore_rag_obs()
    full_obs = f'{rag_name} <- {json.dumps({"q": "x"}, ensure_ascii=False)} => {inner_obs}'

    # 与节点完全同参地预算本步单元，取出稳定的 omitted 编号
    plan_query = " ".join(part for part in (
        "取数1", f"调用 {rag_name} 取数",
        agent_goal_from_state({"intent": IntentContext()}), "查退款政策",
    ) if part)
    pre_units, _, _ = ingest_observation(
        existing_units=[], meta={"next_seq": 1, "rounds": []},
        tool_name=rag_name, round_idx=0, observation=full_obs,
        current_query=plan_query, user_question="查退款政策",
        action_input={"q": "x"}, call_id="plan_0_t1",
    )
    _, omitted = plan_view_uids(pre_units, round_idx=0)
    assert len(omitted) >= 2
    requested = sorted(omitted)[:2]
    unit_text = {u["uid"]: u["text"] for u in pre_units}

    tracer = _RecordingTracer()
    router = _RestoreModelRouter(
        plan_tools=[rag_name],
        subtask_outcomes=[
            {"conclusion": "部分退款政策已看到", "solved": "no",
             "next_action": "continue", "requested_evidence_uids": requested},
            # 恢复步的再提炼：回填已看到；再次索取必须被忽略
            {"conclusion": "回填证据已纳入，退款政策结论完整", "solved": "yes",
             "next_action": "continue", "requested_evidence_uids": ["e1"]},
        ],
        summary_verdicts=[
            {"sufficient": True, "answer": "已完成", "missing_info": "", "suggestion": ""}
        ],
    )
    runner, deps, registry, _ = _build_runner(
        {rag_name: FakeTool(rag_name, inner_obs)},
        _base_config(enable_evidence_board=True),
        model_router=router,
    )
    deps.tracer = tracer

    outcome = await runner.run(
        deps=deps, user_input="查退款政策", session_id="s-evidence-restore",
        mode="plan_execute", intent=IntentContext(),
    )
    assert outcome.paused is False

    # 真实工具只被调用一次：恢复步不经过注册表
    assert [inv["name"] for inv in registry.invocations] == [rag_name]
    # t1 与恢复步各提炼一次
    assert router._subtask_calls == 2

    snapshot = await aget_snapshot(runner._graph, outcome.run_id)
    values = snapshot.values
    results = values.get("subtask_results") or []
    restore_recs = [r for r in results if r.get("tool_name") == "evidence_restore"]
    assert len(restore_recs) == 1
    rec = restore_recs[0]
    assert rec["internal_evidence_restore"] is True
    assert rec["restore_hits"] == 2
    restore_text = rec["observation"]
    for uid in requested:
        assert f"回填 [{uid}｜来源：policy_" in restore_text
        assert unit_text[uid] in restore_text

    # 回填内容不二次入管：证据单元仍是 t1 的 8 条
    assert len(values.get("evidence_units") or []) == len(pre_units)

    # 计划里恰好一个内部恢复步（恢复步的再次请求被忽略，未连环插入）
    restore_task_ids = [t["id"] for t in values.get("plan") or []
                        if str(t.get("id", "")).startswith("correction_evidence_")]
    assert restore_task_ids == ["correction_evidence_t1"]

    # 恢复步的提炼提示词里确实看到了回填原文
    second_distill_blob = json.dumps(router.subtask_distill_messages[1],
                                     ensure_ascii=False, default=str)
    assert "回填 [" in second_distill_blob
    assert unit_text[requested[0]] in second_distill_blob

    # trace：evidence.round 带 exempt 计数；evidence.restore 受理 1 次 +
    # 恢复步再次请求被忽略 1 次
    round_events = [p for e, p in tracer.events if e == "evidence.round"]
    assert round_events and all("exempt" in p for p in round_events)
    restore_events = [p for e, p in tracer.events if e == "evidence.restore"]
    applied = [p for p in restore_events if p.get("applied")]
    skipped = [p for p in restore_events if not p.get("applied")]
    assert len(applied) == 1
    assert applied[0]["accepted"] == requested
    assert applied[0]["rejected"] == []
    assert applied[0]["chars"] > 0
    assert len(skipped) == 1
    assert "每子任务最多一次" in skipped[0]["reason"]


@pytest.mark.asyncio
async def test_plan_evidence_restore_off_when_switch_disabled() -> None:
    """特性开关关闭时：模型即使填了 requested_evidence_uids 也零行为变化。"""
    rag_name = "rag_knowledge_search"
    inner_obs = _restore_rag_obs()
    router = FakeModelRouter(
        plan_tools=[rag_name],
        subtask_outcomes=[
            {"conclusion": "已完成", "solved": "yes", "next_action": "continue",
             "requested_evidence_uids": ["e1", "e2"]},
        ],
        summary_verdicts=[
            {"sufficient": True, "answer": "已完成", "missing_info": "", "suggestion": ""}
        ],
    )
    runner, deps, registry, _ = _build_runner(
        {rag_name: FakeTool(rag_name, inner_obs)},
        _base_config(),  # 不开 enable_evidence_board
        model_router=router,
    )
    outcome = await runner.run(
        deps=deps, user_input="查退款政策", session_id="s-evidence-restore-off",
        mode="plan_execute", intent=IntentContext(),
    )
    assert outcome.paused is False
    assert [inv["name"] for inv in registry.invocations] == [rag_name]
    snapshot = await aget_snapshot(runner._graph, outcome.run_id)
    values = snapshot.values
    assert not values.get("evidence_units")
    assert all("correction_evidence_" not in str(t.get("id"))
               for t in values.get("plan") or [])
