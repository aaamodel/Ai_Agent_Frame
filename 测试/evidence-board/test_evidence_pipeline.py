# -*- coding: utf-8 -*-
"""证据板 T1–T3 单元测试：chunker / 打分去重选择 / fetch / 三视图渲染 / 降级。

直接运行：
    cd 项目根
    python -m pytest 测试/evidence-board/test_evidence_pipeline.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())))

from app.core.agent.graph.deps import GraphDeps
from app.core.agent.evidence import (
    FETCH_TOOL_NAME,
    handle_fetch_evidence,
    ingest_observation,
    render_fc_messages,
    render_text_history_lines,
)
from app.core.agent.evidence.chunkers import (
    TABLE_MAX_ROWS,
    WEB_BLOCK_MAX_CHARS,
    chunk_observation,
)
from app.core.agent.evidence.fetch import (
    fetch_tool_definition,
    is_fetch_call,
    parse_fetch_arguments,
)
from app.core.agent.evidence.models import KIND_STATUS, EvidenceUnit
from app.core.agent.evidence.pipeline import (
    BOARD_BASE_CHARS,
    BOARD_CAP_CHARS,
    BOARD_STEP_CHARS,
    SCORE_MIN,
    board_budget_chars,
    board_top_k,
    is_duplicate,
    select_board,
    simhash,
)
from app.core.agent.toolcall import ToolCall


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
def rag_obs(query: str, entries: List[tuple]) -> str:
    """entries: [(来源, 内容片段), ...]"""
    body = "".join(
        f"[{i + 1}] 来源文献: {src}\n内容片段: {content}\n"
        for i, (src, content) in enumerate(entries)
    )
    return f"--- 知识库检索结果 (查询: {query}) ---\n{body}"


JUNK_LONG = (
    "行政后勤通知：本周食堂菜单安排为周一红烧肉、周二番茄炒蛋、周三清炒时蔬，"
    "班车时刻表现已调整为早七点半从园区南门发车，途经西苑站与北站，"
    "节假日值班排班请各部门于本周五前报备行政组，逾期不再受理。" * 4
)

# 超过短观测豁免阈值（300 字符）的长政策片段：用于需要走切块/打分/模糊判重
# 通道的用例（≤300 字符的观测会被折叠为豁免单元，见 D5a）。
LONG_POLICY = (
    "7天无理由退货需商品完好，退款3个工作日内原路退回付款账户。"
    "退回商品须保持原包装与配件完好，赠品一并寄回，"
    "运费险在退款完成后三个工作日内自动理赔至原支付账户。" * 4
)


def _ingest(units, meta, *, tool_name, round_idx, observation, query,
            user_q="公司退款政策是什么", action_input=None, call_id=None):
    return ingest_observation(
        existing_units=units, meta=meta,
        tool_name=tool_name, round_idx=round_idx, observation=observation,
        current_query=query, user_question=user_q,
        action_input=action_input if action_input is not None else {"query": query},
        call_id=call_id,
    )


class CaptureTracer:
    def __init__(self) -> None:
        self.events: List[tuple] = []

    def log_event(self, trace_id: str, event: str, payload: Any = None) -> None:
        self.events.append((event, payload))


def _deps(tracer: CaptureTracer) -> GraphDeps:
    return GraphDeps(
        config={}, model_router=None, memory=None, tools=None,
        skill_manager=None, tracer=tracer,
    )


# ---------------------------------------------------------------------------
# 1.3/1.4 chunker
# ---------------------------------------------------------------------------
def test_rag_chunker_parses_header_sources_and_refs() -> None:
    obs = rag_obs("退款政策", [("售后制度.docx", "7天无理由退货，3个工作日到账。"),
                              ("物流手册.pdf", "48小时内反馈破损可理赔。")])
    blocks = chunk_observation(tool_name="rag_knowledge_search", observation=obs,
                               action_input={"query": "退款政策"}, call_id="c0")
    assert len(blocks) == 2
    assert [b["source"] for b in blocks] == ["售后制度.docx", "物流手册.pdf"]
    assert blocks[0]["ref"]["rag_index"] == 1
    assert blocks[0]["ref"]["call_id"] == "c0"
    assert blocks[0]["text"].startswith("7天无理由")


def test_rag_empty_and_error_classified() -> None:
    empty = "针对查询项 [xxx] 未匹配到任何知识库文档，请调整关键词后重试。"
    blocks = chunk_observation(tool_name="rag_knowledge_search", observation=empty)
    assert blocks[0]["kind"] == "status"
    err = "rag_knowledge_search 执行期间发生异常错误: 连接超时"
    blocks = chunk_observation(tool_name="rag_knowledge_search", observation=err)
    assert blocks[0]["kind"] == "error"


def test_table_chunker_truncates_rows_and_marks_truncated() -> None:
    rows = "\n".join(f"| {i} | 客户{i} | {i * 100} |" for i in range(1, 41))
    obs = "查询结果如下：\n| id | 客户 | 金额 |\n| --- | --- | --- |\n" + rows
    blocks = chunk_observation(tool_name="sales_sql_query", observation=obs)
    table_blocks = [b for b in blocks if b["kind"] == "table"]
    assert len(table_blocks) == 1
    block = table_blocks[0]
    assert block["truncated"] is True
    assert f"共 {TABLE_MAX_ROWS + 10} 行" in block["text"] or "共 40 行" in block["text"]


def test_web_search_chunker_entries_and_cap() -> None:
    entries = "".join(
        f"[{i}] 标题: 退款新闻标题{i}\n链接: https://example.com/{i}\n"
        f"来源: example.com\n摘要: {'退款时效政策内容' * 60}\n\n"
        for i in range(1, 4)
    )
    obs = f"以下是关于「退款政策」的最新联网搜索结果：\n{entries}"
    blocks = chunk_observation(tool_name="web_search", observation=obs, call_id="w0")
    # 3 个来源；超长摘要允许按 500 字拆成多块
    assert len(blocks) >= 3
    assert {b["ref"].get("web_index") for b in blocks} == {1, 2, 3}
    assert all(b["source"].startswith("https://example.com/") for b in blocks)
    assert all(len(b["text"]) <= WEB_BLOCK_MAX_CHARS for b in blocks)
    # 综合摘要
    obs2 = ("以下是关于「x」的最新联网搜索结果：\n【搜索综合摘要】\n这是综合结论。\n\n"
            "[1] 标题: t\n链接: https://e.com/1\n摘要: s\n")
    blocks2 = chunk_observation(tool_name="tavily_search_internal", observation=obs2)
    assert any(b["source"] == "搜索综合摘要" for b in blocks2)
    # 降级提示
    blocks3 = chunk_observation(
        tool_name="web_search",
        observation="【系统提示】联网搜索工具目前不可用（已尝试 3 次均失败）。",
    )
    assert blocks3[0]["kind"] == KIND_STATUS


def test_feishu_and_kg_chunkers() -> None:
    ok = chunk_observation(
        tool_name="feishu_bitable_tool",
        observation="成功：数据已写入飞书多维表，生成的 Record_ID 为: rec123",
    )
    assert ok[0]["kind"] == KIND_STATUS
    long_text = "图谱实体关系描述。" * 200
    blocks = chunk_observation(tool_name="knowledge_graph_search", observation=long_text)
    assert blocks  # 超长观测必须被切开
    assert all(len(b["text"]) <= WEB_BLOCK_MAX_CHARS for b in blocks)


# ---------------------------------------------------------------------------
# 1.5–1.9 打分 / 去重 / 选择 / 编排
# ---------------------------------------------------------------------------
def test_budget_curve_and_top_k() -> None:
    assert board_budget_chars(0) == BOARD_BASE_CHARS
    assert board_budget_chars(1) == BOARD_BASE_CHARS + BOARD_STEP_CHARS
    assert board_budget_chars(2) == BOARD_BASE_CHARS + 2 * BOARD_STEP_CHARS
    assert board_budget_chars(9) == BOARD_CAP_CHARS
    assert board_top_k(0) == 5
    assert board_top_k(5) == 10
    assert board_top_k(9) == 10


def test_ingest_scores_key_evidence_on_board_junk_low() -> None:
    obs = rag_obs("退款政策", [
        ("售后制度.docx", "7天无理由退货需商品完好，退款3个工作日内原路退回付款账户。"),
        ("物流手册.pdf", JUNK_LONG),
        ("考勤须知.txt", JUNK_LONG.replace("班车", "通勤车")),
    ])
    units, meta, report = _ingest(
        [], {"next_seq": 1, "rounds": []},
        tool_name="rag_knowledge_search", round_idx=0,
        observation=obs, query="退款政策是什么",
    )
    assert report.new == 3
    key = next(u for u in units if u["uid"] == "e1")
    assert key["selected"] is True
    assert key["score"] >= SCORE_MIN
    junk = [u for u in units if u["uid"] in ("e2", "e3")]
    assert all(u["score"] < SCORE_MIN for u in junk)
    assert report.on_board >= 1


def test_dedup_identical_and_minor_edit_keeps_distinct_content() -> None:
    a = EvidenceUnit(uid="e1", tool_name="t", round_idx=0, block_idx=0,
                     text=LONG_POLICY)
    b = EvidenceUnit(uid="e2", tool_name="t", round_idx=1, block_idx=0,
                     text=LONG_POLICY)
    c = EvidenceUnit(uid="e3", tool_name="t", round_idx=1, block_idx=1,
                     text="本月食堂新增麻辣香锅窗口，晚餐供应时间延长至晚八点半。")
    a.simhash = simhash(a.text)
    b.simhash = simhash(b.text)
    c.simhash = simhash(c.text)
    assert is_duplicate(b, a) is True
    assert is_duplicate(c, a) is False

    # 端到端：跨轮相同片段判重，also_from 带来源互证
    obs1 = rag_obs("退款政策", [("售后制度.docx", a.text)])
    units, meta, _ = _ingest([], {"next_seq": 1, "rounds": []},
                             tool_name="rag_knowledge_search", round_idx=0,
                             observation=obs1, query="退款政策")
    obs2 = rag_obs("退款时效", [("售后制度v2.docx", a.text)])
    units, meta, report = _ingest(units, meta,
                                  tool_name="rag_knowledge_search", round_idx=1,
                                  observation=obs2, query="退款时效")
    dup = next(u for u in units if u["uid"] == "e2")
    assert dup["dupe_of"] == "e1"
    assert report.duplicated == 1
    winner = next(u for u in units if u["uid"] == "e1")
    assert "售后制度v2.docx" in winner["also_from"]


def test_no_new_evidence_when_all_duplicates() -> None:
    text = LONG_POLICY
    obs1 = rag_obs("退款政策", [("a.docx", text)])
    units, meta, _ = _ingest([], {"next_seq": 1, "rounds": []},
                             tool_name="rag_knowledge_search", round_idx=0,
                             observation=obs1, query="退款政策")
    obs2 = rag_obs("再查一次", [("b.docx", text)])
    units, meta, report = _ingest(units, meta,
                                  tool_name="rag_knowledge_search", round_idx=1,
                                  observation=obs2, query="退款政策")
    assert report.duplicated == 1
    assert report.no_new_evidence is True


def test_error_unit_always_present_but_not_scored() -> None:
    obs = "sales_sql_query 执行期间发生异常错误: no such table: orders"
    units, _, report = _ingest(
        [], {"next_seq": 1, "rounds": []},
        tool_name="sales_sql_query", round_idx=0, observation=obs,
        query="查订单",
    )
    assert units[0]["kind"] == "error"
    assert units[0]["score"] == 0.0
    assert report.new == 1


# ---------------------------------------------------------------------------
# 3.x fetch
# ---------------------------------------------------------------------------
def test_fetch_definition_and_parse() -> None:
    definition = fetch_tool_definition()
    assert definition["function"]["name"] == FETCH_TOOL_NAME
    assert definition["function"]["parameters"]["required"] == ["uid"]

    uid, window, err = parse_fetch_arguments({"uid": "e3", "window": 2})
    assert (uid, window, err) == ("e3", 2, None)
    uid, window, err = parse_fetch_arguments('{"uid": "e1"}')
    assert (uid, window, err) == ("e1", 0, None)
    assert parse_fetch_arguments({})[2]
    assert parse_fetch_arguments({"uid": "e1", "window": 9})[2]
    assert parse_fetch_arguments("not-json")[2]

    call = ToolCall(tool_name=FETCH_TOOL_NAME, arguments={"uid": "e1"})
    assert is_fetch_call(call) is True
    assert is_fetch_call({"tool_name": "rag_knowledge_search"}) is False


def test_handle_fetch_bad_uid_lists_available() -> None:
    units = [EvidenceUnit(uid="e1", tool_name="t", round_idx=0, block_idx=0,
                          text="内容A").to_dict()]
    text, updated = handle_fetch_evidence(
        {"evidence_units": units, "react_messages": []}, uid="e9", round_idx=1
    )
    assert "未找到证据 e9" in text
    assert "可用编号：e1" in text
    assert updated == units


def test_handle_fetch_window_from_original_message_and_mark() -> None:
    obs = rag_obs("退款政策", [
        ("售后制度.docx", "关键政策原文，退款3个工作日到账，遇节假日顺延。" * 8),
        ("物流手册.pdf", "48小时破损理赔条款，须保留外包装并拍照留证。" * 8),
    ])
    units, meta, _ = _ingest(
        [], {"next_seq": 1, "rounds": []},
        tool_name="rag_knowledge_search", round_idx=0, observation=obs,
        query="退款政策", call_id="call_0_0",
    )
    state = {
        "evidence_units": units,
        "react_messages": [
            {"role": "assistant", "content": None},
            {"role": "tool", "tool_call_id": "call_0_0", "content": obs},
        ],
    }
    text, updated = handle_fetch_evidence(
        state, uid="e2", window=1, round_idx=1
    )
    assert "片段 1/2" in text and "片段 2/2" in text and ">>>" in text
    marked = next(u for u in updated if u["uid"] == "e2")
    assert marked["fetched"] is True
    assert marked["ref"]["fetched_round"] == 1


def test_fetched_unit_forced_only_one_round() -> None:
    key = "7天无理由退货需商品完好，退款3个工作日内原路退回付款账户。"
    obs1 = rag_obs("退款政策", [("a.docx", key), ("b.txt", JUNK_LONG)])
    units, meta, _ = _ingest([], {"next_seq": 1, "rounds": []},
                             tool_name="rag_knowledge_search", round_idx=0,
                             observation=obs1, query="退款政策")
    # 模型在第 1 轮激活取回 e2（junk），标记下一轮强制
    state = {"evidence_units": units, "react_messages": [
        {"role": "tool", "tool_call_id": "c", "content": obs1}]}
    _, units = handle_fetch_evidence(state, uid="e2", round_idx=1)
    board_round1 = {u.uid for u in select_board(
        [EvidenceUnit.from_dict(d) for d in units], 1)}
    assert "e2" in board_round1  # 取回后下一轮强制保留
    # 再来一轮新检索后，强制失效，junk 不再占预算
    obs2 = rag_obs("审批流程", [
        ("c.docx", "超过1000元的退款需财务主管审批，每月5日统一打款。"),
        ("d.txt", JUNK_LONG.replace("红烧肉", "糖醋排骨")),
    ])
    units, meta, _ = _ingest(units, meta,
                             tool_name="rag_knowledge_search", round_idx=2,
                             observation=obs2, query="大额退款审批流程",
                             user_q="公司退款政策是什么")
    board_round2 = {u.uid for u in select_board(
        [EvidenceUnit.from_dict(d) for d in units], 2)}
    assert "e2" not in board_round2


# ---------------------------------------------------------------------------
# 2.1/2.2 视图渲染
# ---------------------------------------------------------------------------
def _fc_messages_fixture(obs0: str, obs1: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_0_0", "function": {"name": "rag_knowledge_search",
                                            "arguments": '{"query":"退款政策"}'}}]},
        {"role": "tool", "tool_call_id": "call_0_0", "content": obs0},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1_0", "function": {"name": "rag_knowledge_search",
                                            "arguments": '{"query":"退款时效"}'}}]},
        {"role": "tool", "tool_call_id": "call_1_0", "content": obs1},
    ]


def test_fc_view_owner_rendering_footer_and_immutable_prefix() -> None:
    key1 = "7天无理由退货需商品完好，退款3个工作日内原路退回付款账户。"
    key2 = "超过1000元的退款需财务主管审批，每月5日统一打款。"
    obs0 = rag_obs("退款政策", [("a.docx", key1), ("j.txt", JUNK_LONG)])
    obs1 = rag_obs("退款时效", [("b.docx", key2), ("k.txt", JUNK_LONG.replace("红烧", "清蒸"))])
    units, meta, _ = _ingest([], {"next_seq": 1, "rounds": []},
                             tool_name="rag_knowledge_search", round_idx=0,
                             observation=obs0, query="退款政策", call_id="call_0_0")
    units, meta, _ = _ingest(units, meta,
                             tool_name="rag_knowledge_search", round_idx=1,
                             observation=obs1, query="大额退款审批", call_id="call_1_0")
    messages = _fc_messages_fixture(obs0, obs1)
    rendered = render_fc_messages(messages, units, meta["rounds"], round_idx=1)
    assert rendered[0] == messages[0] and rendered[1] == messages[1]
    tool_msgs = [m for m in rendered if m["role"] == "tool"]
    assert "[e1｜来源：a.docx]" in tool_msgs[0]["content"]
    # 同一条全文只能出现一次（junk 只给索引行）
    assert tool_msgs[0]["content"].count(JUNK_LONG[:20]) <= 1
    assert "相关性:低" in tool_msgs[0]["content"]
    # 尾注只挂最后一条 tool 消息
    assert "证据索引：共" not in tool_msgs[0]["content"]
    assert "证据索引：共 4 条" in tool_msgs[1]["content"]
    assert "fetch_evidence(e编号)" in tool_msgs[1]["content"]


def test_fc_view_keeps_fetch_message_full() -> None:
    key = "7天无理由退货，3个工作日到账。"
    obs = rag_obs("退款政策", [("a.docx", key)])
    units, meta, _ = _ingest([], {"next_seq": 1, "rounds": []},
                             tool_name="rag_knowledge_search", round_idx=0,
                             observation=obs, query="退款政策", call_id="c0")
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "f0", "function": {"name": FETCH_TOOL_NAME,
                                      "arguments": '{"uid":"e1"}'}}]},
        {"role": "tool", "tool_call_id": "f0", "content": "已为你取回 [e1] 的原文：\n全文…"},
    ]
    rendered = render_fc_messages(messages, units, meta["rounds"], round_idx=1)
    assert rendered[1]["content"] == "已为你取回 [e1] 的原文：\n全文…"


def test_text_view_replaces_observation_and_keeps_fail_line() -> None:
    key = "7天无理由退货，退款3个工作日到账。"
    obs = rag_obs("退款政策", [("a.docx", key), ("j.txt", JUNK_LONG)])
    units, meta, _ = _ingest([], {"next_seq": 1, "rounds": []},
                             tool_name="rag_knowledge_search", round_idx=0,
                             observation=obs, query="退款政策")
    history = [
        f"Step 1\nThought: 查政策\nAction: rag_knowledge_search\nObservation: {obs}\n",
        "Step 2\nThought: 格式错了\n[系统校验 FAIL，结构化回传以便下一轮 Retry]\n"
        "Observation: 【系统校验 FAIL（进入下一轮循环 Retry）】：\n- parse\n",
    ]
    rendered = render_text_history_lines(history, units, meta["rounds"], round_idx=1)
    assert "[e1｜来源：a.docx]" in rendered[0]
    assert "证据索引：共" in rendered[-1]
    # FAIL 块原样保留
    assert "系统校验 FAIL" in rendered[1]


# ---------------------------------------------------------------------------
# 2.7/4.1 降级与 trace
# ---------------------------------------------------------------------------
def test_ingest_fallback_swallows_error_and_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    en = importlib.import_module("app.core.agent.graph.nodes.execute_node")

    tracer = CaptureTracer()
    deps = _deps(tracer)

    def boom(**_kwargs: Any) -> List[Dict[str, Any]]:
        raise RuntimeError("chunk boom")

    monkeypatch.setattr(en, "ingest_observation", boom)
    units, meta = en._evidence_ingest(
        deps, "t1", existing_units=[], meta={}, user_question="q",
        tool_name="rag_knowledge_search", round_idx=0,
        observation="obs", current_query="q", action_input={}, call_id="c",
    )
    assert units is None and meta is None
    events = [e for e in tracer.events if e[0] == "evidence.fallback"]
    assert events and events[0][1]["stage"] == "ingest"


def test_evidence_round_trace_payload_fields() -> None:
    import importlib

    en = importlib.import_module("app.core.agent.graph.nodes.execute_node")

    tracer = CaptureTracer()
    deps = _deps(tracer)
    obs = rag_obs("退款政策", [("a.docx", "7天无理由退货，3个工作日到账。")])
    en._evidence_ingest(
        deps, "t1", existing_units=[], meta={"next_seq": 1, "rounds": []},
        user_question="q", tool_name="rag_knowledge_search", round_idx=0,
        observation=obs, current_query="退款政策",
        action_input={"query": "退款政策"}, call_id="c",
    )
    event, payload = next(e for e in tracer.events if e[0] == "evidence.round")
    for key in ("new", "duplicated", "low_score", "on_board",
                "board_chars", "no_new_evidence"):
        assert key in payload, f"evidence.round 缺字段 {key}"


def test_evidence_fetch_trace_payload_fields() -> None:
    import importlib

    en = importlib.import_module("app.core.agent.graph.nodes.execute_node")

    tracer = CaptureTracer()
    deps = _deps(tracer)
    obs = rag_obs("退款政策", [("a.docx", "7天无理由退货，退款3个工作日到账。")])
    units, _meta = en._evidence_ingest(
        deps, "t1", existing_units=[], meta={"next_seq": 1, "rounds": []},
        user_question="q", tool_name="rag_knowledge_search", round_idx=0,
        observation=obs, current_query="退款政策",
        action_input={"query": "退款政策"}, call_id="c",
    )
    state = {
        "evidence_units": units,
        "react_messages": [
            {"role": "tool", "tool_call_id": "c", "content": obs}],
    }
    # 合法回取：hit=True
    en._evidence_fetch(
        deps, "t1", state=state,
        arguments={"uid": "e1", "window": 0}, next_round=1,
    )
    # 非法参数（缺 uid）：hit=False，uid 置空
    en._evidence_fetch(
        deps, "t1", state=state, arguments={"window": 0}, next_round=1,
    )
    fetch_events = [p for name, p in tracer.events if name == "evidence.fetch"]
    assert len(fetch_events) == 2
    for payload in fetch_events:
        for key in ("uid", "window", "chars", "hit"):
            assert key in payload, f"evidence.fetch 缺字段 {key}"
    assert fetch_events[0]["hit"] is True
    assert fetch_events[0]["uid"] == "e1"
    assert fetch_events[0]["chars"] > 0
    assert fetch_events[1]["hit"] is False
    assert fetch_events[1]["uid"] == ""
