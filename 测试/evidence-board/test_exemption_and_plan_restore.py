# -*- coding: utf-8 -*-
"""T5 单元测试：短观测豁免通道（D5a）+ plan_execute 延迟一步回取（D10）。

覆盖：
- 入管：≤300 字符非 error/status 观测折叠为 1 个豁免单元，不打分、只做
  sha1 精确判重、不占板预算；
- ReAct 视图：3 轮内原文直出，第 4 轮（round_idx 差 ≥3）桩化，桩可经
  fetch_evidence 取回；尾注豁免全文/桩计数；
- plan 视图：≤300 原文直出；超长观测的省略索引含 requested_evidence_uids 措辞；
- 协议：SubTaskOutcomeSchema 新字段默认 None、最多 3 条；
- 节点层：_apply_evidence_restore 受理/拒绝/链式闸门/配额，build_plan_restore_text
  回填（含 truncated 经 subtask_results 重切）。

直接运行：
    cd 项目根
    python -m pytest 测试/evidence-board/test_exemption_and_plan_restore.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent.evidence import (  # noqa: E402
    handle_fetch_evidence,
    ingest_observation,
    render_fc_messages,
    render_plan_observation,
    render_text_history_lines,
)
from app.core.agent.evidence.chunkers import chunk_observation  # noqa: E402
from app.core.agent.evidence.fetch import (  # noqa: E402
    EVIDENCE_RESTORE_TOOL,
    build_plan_restore_text,
)
from app.core.agent.evidence.models import EvidenceUnit  # noqa: E402
from app.core.agent.evidence.pipeline import (  # noqa: E402
    EXEMPT_KEEP_ROUNDS,
    EXEMPT_OBS_CHARS,
    EXEMPT_STUB_HEAD_CHARS,
)
from app.core.agent.evidence.view import plan_view_uids  # noqa: E402
from app.core.agent.graph.nodes.execute_node import (  # noqa: E402
    _apply_evidence_restore,
    _apply_step_correction,
    _restore_uids_from_hint,
)
from app.core.agent.step_correction import Candidate  # noqa: E402
from app.query_intent.llm_schemas import SubTaskOutcomeSchema  # noqa: E402

TOOL = "rag_knowledge_search"


# ── 夹具 ───────────────────────────────────────────────────────────────────
def rag_obs(query: str, entries: List[tuple]) -> str:
    body = "".join(
        f"[{i + 1}] 来源文献: {src}\n内容片段: {content}\n"
        for i, (src, content) in enumerate(entries)
    )
    return f"--- 知识库检索结果 (查询: {query}) ---\n{body}"


def _ingest(units, meta, *, obs, round_idx, query="退款政策", user_q="公司退款政策是什么",
            tool_name=TOOL, call_id=None):
    return ingest_observation(
        existing_units=units, meta=meta,
        tool_name=tool_name, round_idx=round_idx, observation=obs,
        current_query=query, user_question=user_q,
        action_input={"query": query}, call_id=call_id,
    )


def _fresh_meta() -> Dict[str, Any]:
    return {"next_seq": 1, "rounds": []}


# 8 条各自独立、足够长（单条 ≈200 字）且措辞互不相同的片段 → round0 板预算
# （1200 字 / 4 条）装不下，稳定产生 ≥3 个 omitted uid。
# ⚠️ 条目间不能仅差一两个字：SimHash/Jaccard 会把近重复片段判重合并。
_LONG_BODIES = [
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
LONG_ENTRIES = [(f"policy_chapter_{i}.txt", body)
                for i, body in enumerate(_LONG_BODIES, start=1)]
LONG_OBS = rag_obs("退款政策", LONG_ENTRIES)


def _round0_units_with_omissions():
    units, meta, report = _ingest([], _fresh_meta(), obs=LONG_OBS, round_idx=0,
                                  call_id="plan_0_t1")
    shown, omitted = plan_view_uids(units, round_idx=0)
    assert len(omitted) >= 3, f"夹具应产生≥3个省略单元，实际 shown={shown}"
    return units, meta, shown, omitted


class _RecordTracer:
    def __init__(self) -> None:
        self.events: List[tuple] = []

    def log_event(self, trace_id: str, event: str, payload: Any = None) -> None:
        self.events.append((event, payload or {}))


_PLAN = [
    {"id": "t1", "title": "取数1", "action_type": "tool", "tool_name": TOOL},
    {"id": "t2", "title": "取数2", "action_type": "tool", "tool_name": TOOL},
]


# ═══════════════════════════════════════════════════════════════════════════
# 5.1 常量
# ═══════════════════════════════════════════════════════════════════════════
def test_exempt_constants_and_default_field():
    assert EXEMPT_OBS_CHARS == 300
    assert EXEMPT_KEEP_ROUNDS == 3
    assert EXEMPT_STUB_HEAD_CHARS == 60
    unit = EvidenceUnit(uid="e1", tool_name="t", round_idx=0, block_idx=0,
                        text="x")
    assert unit.exempt is False
    assert EvidenceUnit.from_dict(unit.to_dict()).exempt is False


# ═══════════════════════════════════════════════════════════════════════════
# 5.2 入管豁免
# ═══════════════════════════════════════════════════════════════════════════
def test_short_observation_builds_single_unscored_exempt_unit():
    obs = rag_obs("目录", [("a.txt", "文件列表：发票模板.docx、报销指引.pdf，共 2 个文件。")])
    assert len(obs.strip()) <= EXEMPT_OBS_CHARS
    units, _, report = _ingest([], _fresh_meta(), obs=obs, round_idx=0)
    assert len(units) == 1
    unit = units[0]
    assert unit["exempt"] is True
    assert unit["score"] == 0.0
    assert unit["selected"] is False  # 豁免单元不占板预算
    assert report.exempt == 1
    assert report.low_score == 0       # 豁免不计低分
    assert report.no_new_evidence is False


def test_exempt_dedup_is_exact_only():
    text = "文件列表：发票模板.docx、报销指引.pdf，共 2 个文件。"
    obs1 = rag_obs("目录", [("a.txt", text)])
    units, meta, _ = _ingest([], _fresh_meta(), obs=obs1, round_idx=0)
    # 逐字重复（同一目录被反复列举）→ 精确判重命中
    units, meta, report2 = _ingest(units, meta, obs=obs1, round_idx=1)
    dup = next(u for u in units if u["uid"] == "e2")
    assert dup["dupe_of"] == "e1"
    assert report2.duplicated == 1
    assert report2.exempt == 0  # 判重豁免单元不占豁免名额

    # 轻微改写的短观测：不做模糊判重，作为新豁免单元接纳
    edited = "文件列表：发票模板.docx、报销指引.pdf，合计 2 个文件。"
    obs3 = rag_obs("目录", [("a.txt", edited)])
    units, _, report3 = _ingest(units, meta, obs=obs3, round_idx=2)
    assert len(units) == 3
    assert not units[-1]["dupe_of"]
    assert units[-1]["exempt"] is True
    assert report3.duplicated == 0


def test_multichunk_observation_between_200_and_300_still_folded():
    # chunker 按条目切出 ≥2 块、但整条观测仍 ≤300 → 豁免折叠为 1 个单元
    obs = rag_obs("目录", [
        ("a.txt", "一季度发票与签收单存于 finance 目录。"),
        ("b.txt", "二季度对账单由财务主管在每月五日前归档。"),
    ])
    raw_blocks = chunk_observation(tool_name=TOOL, observation=obs,
                                   action_input={"query": "目录"})
    assert len(raw_blocks) >= 2, "夹具前提：chunker 应切出多块"
    assert len(obs.strip()) <= EXEMPT_OBS_CHARS
    units, _, _ = _ingest([], _fresh_meta(), obs=obs, round_idx=0)
    assert len(units) == 1
    assert units[0]["exempt"] is True
    assert obs.strip() in units[0]["text"]


def test_short_error_observation_is_not_exempt():
    obs = "rag_knowledge_search 执行期间发生异常错误: connection timeout"
    units, _, _ = _ingest([], _fresh_meta(), obs=obs, round_idx=0)
    assert len(units) == 1
    assert units[0]["kind"] == "error"
    assert units[0]["exempt"] is False


# ═══════════════════════════════════════════════════════════════════════════
# 5.3 ReAct 视图：3 轮原文，之后桩化，桩可取回
# ═══════════════════════════════════════════════════════════════════════════
def _exempt_round_fixture(obs: str, call_id: str = "call_0_0"):
    units, meta, _ = _ingest([], _fresh_meta(), obs=obs, round_idx=0,
                             call_id=call_id)
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "function": {"name": TOOL,
                                         "arguments": '{"query":"目录"}'}}]},
        {"role": "tool", "tool_call_id": call_id, "content": obs},
    ]
    return units, meta, messages


def test_react_views_keep_full_text_for_three_rounds_then_stub():
    short = "文件列表：发票模板.docx、报销指引.pdf，共 2 个文件。"
    obs = rag_obs("目录", [("a.txt", short)])
    units, meta, messages = _exempt_round_fixture(obs)
    needle = "发票模板.docx"

    for current_round in (0, 1, 2):
        rendered = render_fc_messages(messages, units, meta["rounds"],
                                      round_idx=current_round)
        tool_msg = next(m for m in rendered if m["role"] == "tool")
        assert tool_msg["content"].count(needle) == 1
        assert "短观测全文 1 条/桩 0 条" in tool_msg["content"]

    rendered3 = render_fc_messages(messages, units, meta["rounds"], round_idx=3)
    stub_msg = next(m for m in rendered3 if m["role"] == "tool")
    # 原文不再直出，只剩桩：编号 + 工具名 + 原始字符数 + 首句预览 + 取回提示
    assert stub_msg["content"].count(needle) == 0
    assert "[e1]" in stub_msg["content"]
    assert TOOL in stub_msg["content"]
    assert f"原始 {len(units[0]['text'])} 字符" in stub_msg["content"]
    assert "需要全文可调取证据编号" in stub_msg["content"]
    assert "短观测全文 0 条/桩 1 条" in stub_msg["content"]


def test_exempt_stub_fetchable_via_fetch_evidence():
    short = "文件列表：报销指引.pdf，共 1 个文件，存放于共享盘 finance 根目录。"
    obs = rag_obs("目录", [("a.txt", short)])
    units, _, messages = _exempt_round_fixture(obs)
    # 第 3 轮视图里是桩
    rendered3 = render_fc_messages(messages, units, [], round_idx=3)
    assert "需要全文可调取证据编号" in rendered3[-1]["content"]
    # 桩编号走既有 fetch_evidence 通道取回原文
    state = {"evidence_units": units, "react_messages": messages}
    text, updated = handle_fetch_evidence(state, uid="e1", window=0, round_idx=3)
    assert "报销指引.pdf" in text
    assert next(u for u in updated if u["uid"] == "e1")["fetched"] is True


def test_exempt_duplicate_renders_pointer_not_second_copy():
    short = "文件列表：发票模板.docx、报销指引.pdf，共 2 个文件。"
    obs1 = rag_obs("目录", [("a.txt", short)])
    units, meta, _ = _ingest([], _fresh_meta(), obs=obs1, round_idx=0,
                             call_id="call_0_0")
    obs2 = obs1  # 同一目录被反复列举：逐字重复
    units, meta, _ = _ingest(units, meta, obs=obs2, round_idx=1,
                             call_id="call_1_0")
    history = [
        f"Step 1\nThought: 看目录\nAction: {TOOL}\nObservation: {obs1}\n",
        f"Step 2\nThought: 再看一次\nAction: {TOOL}\nObservation: {obs2}\n",
    ]
    rendered = render_text_history_lines(history, units, meta["rounds"],
                                         round_idx=1)
    blob = "\n".join(rendered)
    # 原文只出现一次；重复豁免单元渲染重复指针
    assert blob.count("发票模板.docx") == 1
    assert "与 [e1] 内容重复" in blob
    assert "短观测全文 1 条/桩 0 条" in blob


# ═══════════════════════════════════════════════════════════════════════════
# 5.4 plan 视图
# ═══════════════════════════════════════════════════════════════════════════
def test_plan_view_short_observation_is_verbatim():
    obs = rag_obs("目录", [("a.txt", "短目录结果，共两个文件：发票模板与报销指引。")])
    assert len(obs.strip()) <= EXEMPT_OBS_CHARS
    units, _, _ = _ingest([], _fresh_meta(), obs=obs, round_idx=0)
    rendered = render_plan_observation(obs, units, round_idx=0)
    assert rendered == obs


def test_plan_view_long_observation_lists_omitted_with_new_wording():
    units, _, shown, omitted = _round0_units_with_omissions()
    rendered = render_plan_observation(LONG_OBS, units, round_idx=0)
    assert "已省略（低相关；如需全文，在 requested_evidence_uids 填入对应编号）" in rendered
    for uid in omitted:
        assert f"[{uid}]" in rendered
    for uid in shown:
        assert f"[{uid}｜" in rendered or f"[{uid}] " in rendered


# ═══════════════════════════════════════════════════════════════════════════
# 5.5 schema
# ═══════════════════════════════════════════════════════════════════════════
def test_subtask_outcome_schema_requested_uids_field():
    schema = SubTaskOutcomeSchema.model_json_schema()
    assert "requested_evidence_uids" in schema["properties"]

    parsed = SubTaskOutcomeSchema(conclusion="结论", solved="yes")
    assert parsed.requested_evidence_uids is None

    ok = SubTaskOutcomeSchema(conclusion="结论", solved="no",
                              requested_evidence_uids=["e3", "e7"])
    assert ok.requested_evidence_uids == ["e3", "e7"]

    with pytest.raises(ValidationError):
        SubTaskOutcomeSchema(conclusion="结论", solved="no",
                             requested_evidence_uids=["e1", "e2", "e3", "e4"])


# ═══════════════════════════════════════════════════════════════════════════
# 5.6 节点层：受理 / 拒绝 / 闸门 / 回填
# ═══════════════════════════════════════════════════════════════════════════
def _deps_with_tracer(tracer: _RecordTracer) -> SimpleNamespace:
    return SimpleNamespace(tracer=tracer)


def test_restore_accepts_two_valid_uids_and_inserts_internal_step():
    units, _, _, omitted = _round0_units_with_omissions()
    state: Dict[str, Any] = {
        "evidence_units": units,
        "subtask_results": [],
        "step_corrections": [],
        "budget": {},
    }
    tracer = _RecordTracer()
    requested = sorted(omitted)[:2]
    update = _apply_evidence_restore(
        state=state, plan=_PLAN, cursor=0, subtask_id="t1",
        requested=requested, unit_dicts=units,
        deps=_deps_with_tracer(tracer), trace_id="tr",
    )

    new_plan = update["plan"]
    inserted = new_plan[1]
    assert inserted["id"] == "correction_evidence_t1"
    assert inserted["tool_name"] == EVIDENCE_RESTORE_TOOL
    assert inserted["action_type"] == "tool"
    assert _restore_uids_from_hint(inserted) == requested
    # 原计划不被替换，仅在 cursor+1 插入
    assert [t["id"] for t in new_plan if not t["id"].startswith("correction_")] == ["t1", "t2"]
    assert len(update["step_corrections"]) == 1

    events = [p for e, p in tracer.events if e == "evidence.restore"]
    assert len(events) == 1
    payload = events[0]
    assert payload["applied"] is True
    assert payload["accepted"] == requested
    assert payload["rejected"] == []
    assert payload["chars"] > 0


def test_restore_rejects_shown_fake_error_and_cross_round_uids_without_inserting():
    units0, meta, shown, omitted = _round0_units_with_omissions()
    # 追加一个跨轮单元
    units1, _, _ = _ingest(units0, meta,
                           obs=rag_obs("再查", [("z.txt", "另一轮的退款补充说明文本。")]),
                           round_idx=1)
    cross_uid = next(u["uid"] for u in units1 if u["round_idx"] == 1)
    error_units, _, _ = _ingest(
        units1, meta,
        obs="rag_knowledge_search 执行期间发生异常错误: timeout",
        round_idx=0,
    )
    error_uid = next(u["uid"] for u in error_units if u["kind"] == "error")

    state = {"evidence_units": error_units, "subtask_results": [],
             "step_corrections": [], "budget": {}}
    tracer = _RecordTracer()
    update = _apply_evidence_restore(
        state=state, plan=_PLAN, cursor=0, subtask_id="t1",
        requested=[sorted(shown)[0], "e999", error_uid, cross_uid],
        unit_dicts=error_units, deps=_deps_with_tracer(tracer), trace_id="tr",
    )
    # 最多校验 3 条；全部非法（shown/伪造/error）→ 不插步
    assert update == {}
    payload = next(p for e, p in tracer.events if e == "evidence.restore")
    assert payload["applied"] is False
    assert set(payload["rejected"]) >= {sorted(shown)[0], "e999", error_uid}


def test_restore_partial_acceptance_keeps_valid_uid_only():
    units, _, _, omitted = _round0_units_with_omissions()
    state = {"evidence_units": units, "subtask_results": [],
             "step_corrections": [], "budget": {}}
    valid = sorted(omitted)[0]
    update = _apply_evidence_restore(
        state=state, plan=_PLAN, cursor=0, subtask_id="t1",
        requested=[valid, "e999"], unit_dicts=units,
        deps=_deps_with_tracer(_RecordTracer()), trace_id="tr",
    )
    inserted = update["plan"][1]
    assert _restore_uids_from_hint(inserted) == [valid]


def test_restore_dedup_and_cap_three_uids():
    units, _, _, omitted = _round0_units_with_omissions()
    state = {"evidence_units": units, "subtask_results": [],
             "step_corrections": [], "budget": {}}
    chosen = sorted(omitted)[:3]
    update = _apply_evidence_restore(
        state=state, plan=_PLAN, cursor=0, subtask_id="t1",
        requested=[chosen[0], chosen[0], chosen[1], chosen[2], "e999"],
        unit_dicts=units, deps=_deps_with_tracer(_RecordTracer()), trace_id="tr",
    )
    assert _restore_uids_from_hint(update["plan"][1]) == chosen


def test_restore_request_from_correction_step_is_ignored():
    units, _, _, omitted = _round0_units_with_omissions()
    state = {"evidence_units": units, "subtask_results": [],
             "step_corrections": [], "budget": {}}
    tracer = _RecordTracer()
    update = _apply_evidence_restore(
        state=state, plan=_PLAN, cursor=1,
        subtask_id="correction_evidence_t1",
        requested=sorted(omitted)[:2], unit_dicts=units,
        deps=_deps_with_tracer(tracer), trace_id="tr",
    )
    assert update == {}
    payload = next(p for e, p in tracer.events if e == "evidence.restore")
    assert payload["applied"] is False
    assert "每子任务最多一次" in payload["reason"]


def test_restore_blocked_when_correction_quota_exhausted():
    units, _, _, omitted = _round0_units_with_omissions()
    state = {
        "evidence_units": units,
        "subtask_results": [],
        "step_corrections": [{"id": 1}, {"id": 2}],  # 未设预算上限时配额=2
        "budget": {},
    }
    update = _apply_evidence_restore(
        state=state, plan=_PLAN, cursor=0, subtask_id="t1",
        requested=sorted(omitted)[:2], unit_dicts=units,
        deps=_deps_with_tracer(_RecordTracer()), trace_id="tr",
    )
    assert update == {}


def test_restore_id_collision_gets_suffix():
    units, _, _, omitted = _round0_units_with_omissions()
    state = {"evidence_units": units, "subtask_results": [],
             "step_corrections": [], "budget": {}}
    plan = [dict(t) for t in _PLAN]
    plan.insert(1, {"id": "correction_evidence_t1", "tool_name": EVIDENCE_RESTORE_TOOL})
    update = _apply_evidence_restore(
        state=state, plan=plan, cursor=0, subtask_id="t1",
        requested=sorted(omitted)[:1], unit_dicts=units,
        deps=_deps_with_tracer(_RecordTracer()), trace_id="tr",
    )
    # 新步插在 cursor+1（既有同名步顺延到 +2），撞名追加 _2
    assert update["plan"][1]["id"] == "correction_evidence_t1_2"
    assert update["plan"][2]["id"] == "correction_evidence_t1"


def test_restore_uids_from_hint_accepts_str_and_dict():
    task = {"tool_args_hint": json.dumps({"uids": ["e3", "e7"]}, ensure_ascii=False)}
    assert _restore_uids_from_hint(task) == ["e3", "e7"]
    assert _restore_uids_from_hint({"tool_args_hint": {"uids": ["e1"]}}) == ["e1"]
    assert _restore_uids_from_hint({"tool_args_hint": "not-json"}) == []
    assert _restore_uids_from_hint({}) == []


def test_step_correction_and_evidence_restore_coexist():
    """selected_alternative_id 与 requested_evidence_uids 同填：
    纠偏步与证据恢复步都插入，原计划不丢，两条留痕各自独立（D10）。"""
    units, _, _, omitted = _round0_units_with_omissions()
    state = {
        "evidence_units": units,
        "subtask_results": [],
        "step_corrections": [],
        "budget": {},
        "active_tool_names": ["web_search", TOOL],
        "extracted_facts": [],
        "user_input": "查退款政策",
    }
    deps = _deps_with_tracer(_RecordTracer())
    candidates = [Candidate(id="tool:web_search",
                            description="尚未尝试的工具：web_search")]
    correction = _apply_step_correction(
        state=state, plan=_PLAN, cursor=0, skipped_ids=[],
        candidates=candidates, selected="tool:web_search",
        deps=deps, trace_id="tr", subtask_id="t1", trigger="证据缺口",
    )
    assert correction, "纠偏应在配额内生效"
    plan_after_correction = correction["plan"]
    assert plan_after_correction[1]["id"] == "correction_t1"

    restore = _apply_evidence_restore(
        state=state, plan=plan_after_correction, cursor=0, subtask_id="t1",
        requested=sorted(omitted)[:2], unit_dicts=units,
        deps=deps, trace_id="tr",
    )
    ids = [t["id"] for t in restore["plan"]]
    assert ids[0] == "t1" and ids[-1] == "t2"
    assert "correction_t1" in ids and "correction_evidence_t1" in ids

    # 两路留痕在 state reducer 合并后并存、互不覆盖
    records = list(correction["step_corrections"]) + list(restore["step_corrections"])
    triggers = {r.get("trigger") for r in records}
    assert {"证据缺口", "evidence_restore"} <= triggers


def test_restore_step_is_chain_gated_against_step_correction():
    """恢复步 id 以 correction_ 开头：即使模型同时选了替代方向，
    也不得再插纠偏步（复用链式闸门）。"""
    state = {
        "subtask_results": [], "step_corrections": [], "budget": {},
        "active_tool_names": ["web_search"], "extracted_facts": [],
        "user_input": "查退款政策",
    }
    candidates = [Candidate(id="tool:web_search",
                            description="尚未尝试的工具：web_search")]
    update = _apply_step_correction(
        state=state, plan=_PLAN, cursor=1, skipped_ids=[],
        candidates=candidates, selected="tool:web_search",
        deps=_deps_with_tracer(_RecordTracer()), trace_id="tr",
        subtask_id="correction_evidence_t1", trigger="证据缺口",
    )
    assert update == {}


def test_build_plan_restore_text_refills_units_with_headers():
    units, _, _, omitted = _round0_units_with_omissions()
    wanted = sorted(omitted)[:2]
    by_uid = {u["uid"]: u for u in units}
    state = {"evidence_units": units, "subtask_results": []}
    text, hits = build_plan_restore_text(state, wanted + ["e999"])
    assert hits == 2
    for uid in wanted:
        assert f"[{uid}" in text
        assert by_uid[uid]["text"] in text
    assert "e999" in text and "未找到" in text

    text0, hits0 = build_plan_restore_text(state, ["e999"])
    assert hits0 == 0 and "未找到该证据编号" in text0


def test_build_plan_restore_text_rechunks_truncated_unit_from_subtask_results():
    # truncated 单元的 Unit.text 只是残片；回填须经 ref.call_id 从
    # subtask_results 的已保存原始观测重切取完整块。
    truncated = EvidenceUnit(
        uid="e20", tool_name=TOOL, round_idx=0, block_idx=1,
        text="旧截断残片不应出现在回填里",
        source="doc2.txt", ref={"call_id": "plan_0_t1"},
        kind="content", truncated=True,
    ).to_dict()
    rec_obs = LONG_OBS
    state = {
        "evidence_units": [truncated],
        "subtask_results": [{"subtask_id": "t1", "observation": rec_obs}],
    }
    blocks = chunk_observation(tool_name=TOOL, observation=rec_obs,
                               action_input=None, call_id="plan_0_t1")
    assert len(blocks) >= 2
    text, hits = build_plan_restore_text(state, ["e20"])
    assert hits == 1
    assert "旧截断残片" not in text
    assert str(blocks[1].get("text") or "") in text
    assert "回填 [e20" in text
