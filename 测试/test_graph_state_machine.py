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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    ) -> None:
        """``subtask_outcomes``：脚本化「子任务提炼」的控制协议返回，
        用于验证跳过 / 提前收尾是否真的让后续子任务不再执行。"""
        self._react_tool = react_tool
        self._react_args = react_args or {"q": "北京天气"}
        self._plan_tools = list(plan_tools if plan_tools is not None else ["echo_tool"])
        self._summary_verdicts = list(summary_verdicts or self.DEFAULT_VERDICTS)
        self._plan_calls = 0
        self._summary_calls = 0
        self.summary_call_count = 0  # 对外只读：实际发生的 summarize 调用次数
        self._subtask_calls = 0
        self._subtask_outcomes = list(subtask_outcomes or [])

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
            return SimpleNamespace(
                content="", reasoning_content="",
                tool_calls=[{"id": "fc-plan-1", "function": {
                    "name": chosen,
                    "arguments": json.dumps({"q": "x"}, ensure_ascii=False),
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

    # plan 跑完 + L2/L3 证据不足信号 + 有次数与预算余量 → replan
    assert route_after_execute({
        "final_answer": "", "plan": [{"id": "a"}], "cursor": 1,
        "insufficiency_signal": "sig",
        "replan_attempts": 0, "max_replan": 2,
        "budget": {"total_budget": 20, "total_used": 1},
    }) == "replan"

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
        "insufficiency_signal": "sig",
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
    # L3 证据不足且有余量 → replan
    assert route_after_summarize({
        "insufficiency_signal": "sig",
        "replan_attempts": 0, "max_replan": 2,
        "budget": {"total_budget": 20, "total_used": 3},
    }) == "replan"
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
    assert outcome.response.degraded is True
    assert "暂无记录" in outcome.response.answer
    # 取数工具只调用一次（空数据记账后计划收尾；L2 判全坏 → replan 纯推理，无新工具）
    assert len(registry.invocations) == 1
    # 初始 plan + 1 次 replan；summarize 只调 1 次（replan 后纯推理被判定 sufficient）
    assert deps.model_router._plan_calls == 2
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
            {"sufficient": False, "answer": "草稿：只有无关信息",
             "missing_info": "缺少退货率数据", "suggestion": "改用售后类数据源"},
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
