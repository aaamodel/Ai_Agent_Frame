# -*- coding: utf-8 -*-
"""证据板 4.3/4.4 验收夹具：3 轮 RAG（每轮 5 条，2 关键 + 3 噪声长片段）。

验收点：
- 4.3 开关关闭固化末轮观测负载基线（发送视图中 tool 消息内容的总字符数，
  作为 CJK 输入 token 的确定性代理口径，注释固化在断言处），开关打开后
  ≤ 基线 40%，并在测试输出中打印前后数值；
- 4.4 标注的 5 条关键证据 uid 在末轮发送视图中全部在板、原文可见；
- 开/关两种模式最终答案行为一致（模型都能基于关键证据给出同一答案）；
- 4.1 trace：开关打开时 3 次 evidence.round 且字段完整，关闭时无该事件。

运行：
    cd 项目根
    python -m pytest 测试/evidence-board/test_evidence_board_fixture.py -q -s
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())))

from langgraph.checkpoint.memory import InMemorySaver

from app.core.agent.graph.builder import compile_agent_graph
from app.core.agent.graph.deps import GraphDeps
from app.core.agent.graph.runner import GraphRunner
from app.core.agent.orchestrator import IntentContext

# ---------------------------------------------------------------------------
# 夹具：3 轮 × 5 片段（关键片段短而密，噪声片段长而无关——贴近真实 RAG 形态）
# uid 按入管顺序编号：第 r 轮第 i 条（均 0 基）=> e{r*5+i+1}
# ---------------------------------------------------------------------------
USER_QUESTION = "公司的退款政策、到账时效和大额退款审批是怎么规定的"

QUERIES = [
    "退款政策和退货条件",
    "退款到账时效多久",
    "大额退款审批流程",
]

# 标注的 5 条关键证据：r0 第 1/5 条、r1 第 1/5 条、r2 第 1 条
KEYS: Dict[str, str] = {
    "e1": "根据退款政策，商品保持完好且在签收后7天内申请的，可走无理由退货通道，退款按原支付路径退回。",
    "e5": "退款到账渠道限定为原支付账户：银行卡原路退回、微信与支付宝按支付流水号退回，不支持跨渠道退款。",
    "e6": "退款到账时效：仓库验收无误后3个工作日内完成退款打款，遇法定节假日则顺延至节后第一个工作日。",
    "e10": "若退款超过5个工作日未到账，可凭退款单号在客服中心查询进度，银行侧处理延迟不超过2个工作日。",
    "e11": "单笔退款金额超过1000元的，须经财务主管审批后由出纳在每月5日、20日两个统一打款日处理。",
}

# 每条约 250~300 字的行政后勤噪声，刻意不含"退款/退货/审批/时效"等查询词
_JUNK_TEXTS = [
    "行政后勤通知：本周食堂菜单安排为周一红烧肉、周二番茄炒蛋、周三清炒时蔬与清蒸鲈鱼，"
    "晚餐窗口延长至八点半；班车时刻表现已调整为早七点半从园区南门发车，途经西苑站与北站，"
    "节假日值班排班请各部门于本周五前报备行政组，逾期不再受理，临时加班乘车需提前一天预约；"
    "另请各部门统计本季度办公用品需求，统一由前台汇总后向供应商集中采购；"
    "宿舍区洗衣机与烘干机已完成季度消毒，公共厨房冰箱将于月底统一清理，请贴好姓名标签。",
    "园区绿化养护简报：本月完成主干道乔木修剪三千平方米，更换时令花卉四万盆，"
    "人工湖水质净化设备运行正常，锦鲤观赏区每日上午十点由专人投喂；地下车库 B 区照明"
    "改造已进入验收阶段，剩余 C 区将于下月中旬动工，请各部门同事将非机动车停放到"
    "指定车棚，严禁占用消防通道，违者将由安保部统一清运并通报所在部门负责人；"
    "垃圾分类督导岗本月增设晚班，大件家具清运需在物业小程序提前两天预约并登记。",
    "工会活动通知：秋季趣味运动会定于本月最后一个周六在北区体育场举办，"
    "项目包括拔河、跳绳、两人三足与趣味投篮与定点飞镖，各部门请于周三下班前把参赛"
    "名单报工会委员会，活动当天提供午餐包与纪念 T 恤一件；读书会秋季书目已更新，"
    "可在前台扫码登记借阅，借阅期限为三十天，逾期未还将暂停下次借阅资格；"
    "职工子女托管班秋季名额尚有富余，报名材料需含户口本复印件与近期一寸照片两张。",
]


def _entries(round_idx: int) -> List[Tuple[str, str]]:
    if round_idx == 0:
        return [
            ("售后制度-退货章节.docx", KEYS["e1"]),
            ("后勤周报-第31期.md", _JUNK_TEXTS[0]),
            ("园区养护简报.docx", _JUNK_TEXTS[1]),
            ("工会秋季活动.pdf", _JUNK_TEXTS[2]),
            ("财务收付渠道说明.xlsx", KEYS["e5"]),
        ]
    if round_idx == 1:
        return [
            ("财务收付渠道说明.xlsx", KEYS["e6"]),
            ("后勤周报-第32期.md", _JUNK_TEXTS[1]),
            ("工会招新通知.pdf", _JUNK_TEXTS[2]),
            ("食堂满意度调查.docx", _JUNK_TEXTS[0]),
            ("客服工单处理规范.docx", KEYS["e10"]),
        ]
    return [
        ("资金审批制度-2024.pdf", KEYS["e11"]),
        ("园区停车管理办法.docx", _JUNK_TEXTS[1]),
        ("食堂菜单公示.pdf", _JUNK_TEXTS[0]),
        ("读书会章程.md", _JUNK_TEXTS[2]),
        ("绿植领养活动通知.docx", _JUNK_TEXTS[2].replace("工会", "团委")),
    ]


def rag_fixture_observation(query: str) -> str:
    round_idx = QUERIES.index(query)
    body = "".join(
        f"[{i + 1}] 来源文献: {src}\n内容片段: {content}\n"
        for i, (src, content) in enumerate(_entries(round_idx))
    )
    return f"--- 知识库检索结果 (查询: {query}) ---\n{body}"


class RagFixtureTool:
    """返回**原始** RAG 观测（不经 FakeTool 的 name<-args=> 包装前缀）。"""

    name = "rag_knowledge_search"
    description = "知识库检索"
    SYSTEM_PROMPT = ""
    parameters: List[Any] = []

    def schema_parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string"}}}

    async def execute(self, **kwargs: Any) -> str:
        return rag_fixture_observation(str(kwargs.get("query")))


class RagRegistry:
    def __init__(self) -> None:
        self.invocations: List[Dict[str, Any]] = []
        self._tool = RagFixtureTool()

    def list_tool_names(self) -> List[str]:
        return ["rag_knowledge_search"]

    def get_tool(self, name: str) -> Optional[RagFixtureTool]:
        return self._tool if name == "rag_knowledge_search" else None

    async def invoke(self, name: str, arguments: Dict[str, Any]) -> str:
        self.invocations.append({"name": name, "arguments": arguments})
        return await self._tool.execute(**(arguments or {}))


_FINAL_ANSWER = "Final Answer: 已掌握全部退款规则（政策、时效、审批）。"
_INSUFFICIENT = "Final Answer: 证据不足，无法完整回答。"
_SUMMARY_ANSWER = "已掌握全部退款规则（政策、时效、审批）。"


class ScriptedRagRouter:
    """依次发起 3 轮 RAG；末轮按发送视图是否含全部关键证据决定答案。"""

    def __init__(self) -> None:
        self.sent_views: List[List[Dict[str, Any]]] = []
        self.sent_tools: List[List[Any]] = []

    async def chat(self, messages: Any, *, purpose_hint: str = "", **kwargs: Any) -> Any:
        system_text = messages[0].get("content", "") if messages else ""
        if "最终总结助手" in system_text:
            return SimpleNamespace(
                content=json.dumps({
                    "sufficient": True, "answer": _SUMMARY_ANSWER,
                    "missing_info": "", "suggestion": "",
                }, ensure_ascii=False),
                reasoning_content="",
            )
        raise AssertionError(f"夹具未预期的 chat 调用: {system_text[:80]}")

    async def chat_with_tools(
        self, messages: Any, tools: Any, tool_choice: Any = None, *,
        purpose_hint: str = "", **kwargs: Any,
    ) -> Any:
        view = [dict(m) for m in messages]
        self.sent_views.append(view)
        self.sent_tools.append(list(tools))
        tool_rounds = sum(1 for m in view if (m or {}).get("role") == "tool")
        if tool_rounds < 3:
            return SimpleNamespace(
                content="", reasoning_content="",
                tool_calls=[{
                    "id": f"rag-call-{tool_rounds}",
                    "function": {
                        "name": "rag_knowledge_search",
                        "arguments": json.dumps(
                            {"query": QUERIES[tool_rounds]}, ensure_ascii=False
                        ),
                    },
                }],
            )
        blob = json.dumps(view, ensure_ascii=False)
        all_present = all(sentence in blob for sentence in KEYS.values())
        return SimpleNamespace(
            content=_FINAL_ANSWER if all_present else _INSUFFICIENT,
            reasoning_content="", tool_calls=[],
        )


class CaptureTracer:
    def __init__(self) -> None:
        self.events: List[Tuple[str, Any]] = []

    def new_trace_id(self) -> str:
        return "trace-evi"

    def start_span(self, name: str, trace_id: str, attributes: Any = None) -> Any:
        return SimpleNamespace(name=name)

    def end_span(self, span: Any, error: Any = None) -> None:
        return None

    def log_event(self, trace_id: str, event: str, payload: Any = None) -> None:
        self.events.append((event, payload))


class _FakeMemory:
    async def get_context(self, *a: Any, **k: Any) -> Any:
        return SimpleNamespace(short_term_messages=[], long_term_items=[])

    async def append_turn(self, *a: Any, **k: Any) -> None:
        return None


class _FakeSkills:
    def __init__(self) -> None:
        self.state = SimpleNamespace(available_skills={})

    async def scan_and_refresh_skills(self) -> None:
        return None

    def resolve_relevant_skill(self, user_input: str) -> Dict[str, Any]:
        return {}


def _base_config(**overrides: Any) -> Dict[str, Any]:
    cfg = {"max_replan_attempts": 1, "react_max_steps": 8,
           "enable_skill_tool_gating": False, "enable_empty_result_replan": False}
    cfg.update(overrides)
    return cfg


async def _run_session(config: Dict[str, Any], session_id: str) -> tuple:
    saver = InMemorySaver()
    graph = compile_agent_graph(saver)
    runner = GraphRunner(graph, saver)
    registry = RagRegistry()
    model_router = ScriptedRagRouter()
    deps = GraphDeps(
        config=config, model_router=model_router, memory=_FakeMemory(),
        tools=registry, skill_manager=_FakeSkills(), tracer=CaptureTracer(),
    )
    outcome = await runner.run(
        deps=deps, user_input=USER_QUESTION, session_id=session_id,
        mode="react", intent=IntentContext(),
    )
    return outcome, model_router, registry, deps


def _tool_payload_chars(view: List[Dict[str, Any]]) -> int:
    """口径：末轮发送视图中全部 tool 角色消息 content 的字符总数。

    项目内没有中文分词级 token 统计器；CJK 场景下字符数与 token 数近似线性，
    作为确定性代理足以固化"≤ 基线 40%"的对比结论。"""
    return sum(len(str(m.get("content") or "")) for m in view if m.get("role") == "tool")


async def _both_sessions() -> tuple:
    off_outcome, off_router, _, _ = await _run_session(
        _base_config(), "fixture-off"
    )
    on_outcome, on_router, _, _ = await _run_session(
        _base_config(enable_evidence_board=True), "fixture-on"
    )
    return off_outcome, off_router, on_outcome, on_router


@pytest.mark.asyncio
async def test_board_cuts_final_round_payload_to_40_percent() -> None:
    # 4.3 末轮观测负载：开关关（位置型硬截断全文）vs 开关开（证据板选择）
    off_outcome, off_router, on_outcome, on_router = await _both_sessions()
    assert off_outcome.response.success
    assert on_outcome.response.success

    # 末轮发送视图 = 最后一次 chat_with_tools（看到 3 条 tool 消息、给 Final Answer）
    off_chars = _tool_payload_chars(off_router.sent_views[-1])
    on_chars = _tool_payload_chars(on_router.sent_views[-1])

    print(f"\n[证据板负载] 开关关={off_chars} 字符, 开关开={on_chars} 字符, "
          f"比例={on_chars / max(off_chars, 1):.1%}")
    # 基线固化：3 轮 × 5 条，3 条噪声长片段约 220 字，现状最近 3 轮近乎全文，
    # 实测末轮 tool 负载 ≈ 2672 字符；夹具漂移时该阈值会先报警。
    assert off_chars >= 2500
    assert on_chars <= off_chars * 0.40


@pytest.mark.asyncio
async def test_key_evidence_on_board_answer_parity_and_trace() -> None:
    off_outcome, off_router, _, _ = await _run_session(_base_config(), "keys-off")
    on_outcome, on_router, on_registry, on_deps = await _run_session(
        _base_config(enable_evidence_board=True), "keys-on"
    )

    # 注册表恰好 3 次真实 RAG 调用
    assert len(on_registry.invocations) == 3

    # 4.4 标注的 5 条关键证据：末轮板头 + 原文都在发送视图中
    final_blob = json.dumps(on_router.sent_views[-1], ensure_ascii=False)
    for uid, sentence in KEYS.items():
        assert f"[{uid}｜来源" in final_blob, f"关键证据 {uid} 未在末轮板上"
        assert sentence in final_blob, f"关键证据 {uid} 原文缺失"
    # 噪声正文不得上板（低相关只允许以短索引行出现，不含正文句子）
    assert "趣味运动会定于本月最后一个周六" not in final_blob
    assert "乔木修剪三千平方米" not in final_blob

    # 行为一致：两模式模型都能看到全部关键证据，给出同一最终答案
    assert off_outcome.response.answer == on_outcome.response.answer
    assert _FINAL_ANSWER in on_outcome.response.answer

    # 4.1 trace：开关打开 → 3 次 evidence.round，字段完整
    round_payloads = [p for name, p in on_deps.tracer.events
                      if name == "evidence.round"]
    assert len(round_payloads) == 3
    for payload in round_payloads:
        for key in ("new", "duplicated", "low_score", "on_board",
                    "board_chars", "no_new_evidence"):
            assert key in payload
