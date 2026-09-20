# -*- coding: utf-8 -*-
"""增量 JSON 字段抽取器单测。

对应 specs Review Focus #4：UTF-8 多字节 / JSON 转义在 chunk 边界被切开。
"""

from app.core.agent.delta_extract import IncrementalJsonFieldExtractor


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
