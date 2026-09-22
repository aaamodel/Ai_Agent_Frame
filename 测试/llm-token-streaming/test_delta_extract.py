# -*- coding: utf-8 -*-
"""增量 JSON 字段抽取器单测。

对应 specs Review Focus #4：UTF-8 多字节 / JSON 转义在 chunk 边界被切开。
"""

from app.core.agent.delta_extract import (
    DisplayRouter,
    IncrementalJsonFieldExtractor,
)


def feed_all(extractor, text, chunk_size):
    """按固定长度切片喂入，返回累计可见文本。"""
    out = []
    for i in range(0, len(text), chunk_size):
        out.append(extractor.feed(text[i:i + chunk_size]))
    return "".join(out)


def test_field_not_present_yet_returns_empty():
    ex = IncrementalJsonFieldExtractor("rewritten_question")
    assert ex.feed('{"sub_questions":') == ""
    assert ex.feed('["a"]}') == ""


def test_basic_extraction():
    ex = IncrementalJsonFieldExtractor("rewritten_question")
    text = '{"rewritten_question":"我们优先做政企","sufficient":true}'
    assert feed_all(ex, text, 4) == "我们优先做政企"
    assert ex.visible_value() == "我们优先做政企"


def test_incremental_returns_only_new_text():
    """feed 的返回值必须**只是新增部分**，否则前端会重复拼接。"""
    ex = IncrementalJsonFieldExtractor("answer")
    assert ex.feed('{"answer":"第一段') == "第一段"
    assert ex.feed('第二段"') == "第二段"
    assert ex.visible_value() == "第一段第二段"


def test_value_not_closed_returns_partial_but_never_trailing_escape():
    ex = IncrementalJsonFieldExtractor("answer")
    got = ex.feed('{"answer":"abc\\')  # 结尾是一个转义引导符
    assert got == "abc"  # 反斜杠本身不能露出来
    got2 = ex.feed('nd")')  # 下一片补上 n → 完整值 "abc\nd"
    # ⚠️ 修正（Ruling，见账本）：原计划断言 "\n"，漏掉了换行后面的 'd'。
    # 拼接后的 JSON 是 {"answer":"abc\nd")，值就是 abc + 换行 + d。
    assert got2 == "\nd"
    assert ex.visible_value() == "abc\nd"


def test_escapes():
    ex = IncrementalJsonFieldExtractor("answer")
    text = '{"answer":"引号\\" 反斜杠\\\\ 换行\\n 制表\\t 星号\\u002a"}'
    assert feed_all(ex, text, 3) == '引号" 反斜杠\\ 换行\n 制表\t 星号*'


def test_field_name_appearing_inside_value_is_not_treated_as_key():
    """值里出现同名字符串时不能重新开始抽取。"""
    ex = IncrementalJsonFieldExtractor("answer")
    text = '{"answer":"前面 answer 后面"}'
    assert feed_all(ex, text, 2) == "前面 answer 后面"


def test_chinese_split_across_chunks():
    """Review Focus #4：一个中文字被切到两个 chunk。"""
    ex = IncrementalJsonFieldExtractor("rewritten_question")
    text = '{"rewritten_question":"政企合作"}'
    # 每个字符单独喂一次，模拟最碎的分片
    assert "".join(ex.feed(c) for c in text) == "政企合作"


def test_stops_at_closing_quote():
    ex = IncrementalJsonFieldExtractor("answer")
    got = feed_all(ex, '{"answer":"abc","other":"xyz"}', 5)
    assert got == "abc"  # 不能把 other 的值也带出来


# ---------------- 改写阶段：一定是 JSON ----------------


def test_rewrite_phase_extracts_field():
    # ⚠️ 线上 AgentRewriteIntentCombinedSchema 的 JSON 字段名是 `rewrite`
    #    （DTO 层才映射成 rewritten_question）。用真实字段名钉住，
    #    抽错成不存在的字段会静默吞掉整个改写阶段的逐字输出。
    r = DisplayRouter("rewrite")
    text = (
        '{"rewrite":"政企优先","agent_goal":"给结论","should_split":false,'
        '"sub_questions":[],"intent_classifications":[]}'
    )
    assert "".join(r.feed(text[i:i + 4]) for i in range(0, len(text), 4)) == "政企优先"


def test_rewrite_router_reset_between_attempts():
    """重试/换候选后：旧尝试的半截 JSON 不能与新尝试拼成脏文本。"""
    r = DisplayRouter("rewrite")
    out1 = "".join(r.feed(c) for c in '{"rewrite":"政企')
    assert out1 == "政企"
    r.reset()
    text2 = '{"rewrite":"医疗优先","should_split":false}'
    out2 = "".join(r.feed(text2[i:i + 3]) for i in range(0, len(text2), 3))
    assert out2 == "医疗优先", "复位后应只显示新尝试的字段值"


def test_rewrite_fenced_json_extracts_field_across_chunks():
    """智谱 glm-4.7 实测流式形态：JSON 外包 ```json 围栏，且围栏三片分开到。

    不剥围栏时首字符不是 `{`，分流器永久 undecided → 整段静默（线上事故根因）。
    """
    r = DisplayRouter("rewrite")
    chunks = [
        "```", "json", '\n{\n  "rewrite": "采购',
        "审批系统", "支持手机端", "提单吗？",
        '",\n  "intent_classifications": []\n}\n', "```",
    ]
    assert "".join(r.feed(c) for c in chunks) == "采购审批系统支持手机端提单吗？"
    assert r.mode == "json"


def test_rewrite_fence_token_alone_stays_undecided():
    """只到 ``` 、换行还没来：不能急着判 plain 或 json（围栏里可能是代码）。"""
    r = DisplayRouter("rewrite")
    assert r.feed("```") == ""
    assert r.mode == "undecided"


def test_answer_fenced_summary_schema_extracts_answer():
    """answer 阶段汇总 JSON 同样可能裹围栏（含 sufficient 探针 → json 抽 answer）。"""
    r = DisplayRouter("answer")
    chunks = [
        "```", "JSON", '\n{"sufficient":true,"answer":"最终',
        "结论", '"}', "\n```",
    ]
    assert "".join(r.feed(c) for c in chunks) == "最终结论"
    assert r.mode == "json"


def test_answer_fenced_non_json_code_falls_back_to_plain():
    """围栏里是普通代码而非 JSON 对象：不得误锁 json，代码文本照常可见。"""
    r = DisplayRouter("answer")
    out = "".join(r.feed(c) for c in ["```py", "thon\n", "print(1)"])
    assert r.mode == "plain"
    assert out.startswith("```python")
    assert "print(1)" in out


def test_answer_router_reset_clears_mode_lock():
    """旧尝试以 { 开头把分流器锁进 json 模式；复位后新尝试的纯文本必须能直出。"""
    r = DisplayRouter("answer")
    r.feed('{"sufficient":true,"answer":"半截')
    assert r.mode == "json"
    r.reset()
    assert r.mode == "undecided"
    assert r.visible_any() is False
    assert r.feed("纯文本新尝试") == "纯文本新尝试"


# ---------------- 答案阶段：四条判定分支 ----------------


def test_answer_plain_text_streams_directly():
    """FC 协议的普通答案：纯文本直出。"""
    r = DisplayRouter("answer")
    assert r.feed("政企合作") == "政企合作"
    assert r.feed("优先") == "优先"


def test_answer_json_looking_but_not_summary_schema_is_plain():
    """Review Focus #2：用户要 JSON 输出时，这是正常答案，不能整段不显示。

    ⚠️ 文本必须长于 200 字符：spec §4.5 规定看到 `{` 后要满 200 字符窗口
    才能排除汇总 schema（`SummaryVerdictSchema`）。
    """
    r = DisplayRouter("answer")
    text = '{"industries":["政企","医疗"],"priority":"high","note":"' + "说明" * 120 + '"}'
    assert len(text) > 200
    got = "".join(r.feed(text[i:i + 5]) for i in range(0, len(text), 5))
    assert got == text, "以 { 开头但不是汇总 schema 的答案必须原样显示"
    assert r.mode == "plain"


def test_answer_summary_schema_is_extracted_not_shown_raw():
    """汇总的结构化输出：只显示 answer 字段，JSON 结构不能漏出来。"""
    r = DisplayRouter("answer")
    text = '{"sufficient":true,"answer":"最终答案在这里","missing_info":""}'
    got = "".join(r.feed(text[i:i + 6]) for i in range(0, len(text), 6))
    assert got == "最终答案在这里"
    assert "sufficient" not in got


def test_answer_react_draft_is_suppressed():
    """Review Focus #3：Thought/Action 是内部草稿，绝不能出现在界面上。"""
    r = DisplayRouter("answer")
    draft = "Thought: 我需要先查一下\nAction: sales_sql_query\nAction Input: {}"
    got = "".join(r.feed(draft[i:i + 8]) for i in range(0, len(draft), 8))
    assert got == ""
    assert r.mode == "suppress"


def test_answer_final_answer_marker_shows_only_tail():
    r = DisplayRouter("answer")
    text = "Thought: 想好了\nFinal Answer: 政企优先，其次医疗"
    got = "".join(r.feed(text[i:i + 7]) for i in range(0, len(text), 7))
    assert got == "政企优先，其次医疗"
    assert "Thought" not in got


def test_answer_typing_before_decision_emits_nothing_but_does_not_lose_text():
    """判定未定时不输出；判定为 plain 后，之前的文本必须补出来，不能丢。"""
    r = DisplayRouter("answer")
    head = '{"industries":["政企","医疗"],"priority":"high"'
    assert r.feed(head) == ""  # 200 字符窗口未满 → 判定未定，不输出
    assert r.feed("}") == ""  # 仍未满
    filler = "x" * 200
    out = r.feed(filler)  # 满 200 → 判定 plain，之前攒下的全部补出
    assert out == head + "}" + filler
    assert r.mode == "plain"


def test_visible_any_reflects_emitted_output():
    """供 SSE 层判断"换候选要不要插废弃标注"。"""
    r = DisplayRouter("answer")
    assert r.visible_any() is False
    r.feed("政企")
    assert r.visible_any() is True


def test_unknown_phase_rejected():
    import pytest

    with pytest.raises(ValueError):
        DisplayRouter("thinking")
