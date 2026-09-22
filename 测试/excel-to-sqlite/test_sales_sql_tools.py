# -*- coding: utf-8 -*-
"""销售 SQL 工具的专项测试（只读护栏 + 受约束写入 + 初始化失败可见）。

覆盖 `agent/sql-query-tool` 的只读与写入护栏部分。
需要真实模型通道的用例（端到端问答）不放在这里，由 `test_sales_sql_e2e.py` 承担。
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.sales_db.schema import create_all  # noqa: E402
from app.core.tools.builtin import sql_vanna as sql_vanna_mod  # noqa: E402
from app.core.tools.builtin.sql_vanna import (  # noqa: E402
    SalesSqlQueryTool,
    SalesSqlWriteTool,
    SalesVanna,
    _assert_readonly,
    get_vanna,
    reset_vanna,
)


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    """临时业务库：只插 3 行线索，够覆盖命中/未命中/多命中。"""
    target = tmp_path / "probe.db"
    conn = sqlite3.connect(str(target))
    create_all(conn)
    conn.executemany(
        "INSERT INTO 线索 (线索编号, 公司全称, 行业, 区域, 负责人) VALUES (?,?,?,?,?)",
        [("LD-001", "甲公司", "制造业", "华北", "王强"),
         ("LD-002", "乙公司", "制造业", "华东", "李娜"),
         ("LD-003", "丙公司", "金融服务", "华南", "张伟")],
    )
    conn.commit()
    monkeypatch.setenv("SALES_DB_PATH", str(target))
    yield conn
    conn.close()


# ── 只读护栏 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("sql", [
    "UPDATE 线索 SET 负责人='x'",
    "DELETE FROM 线索 WHERE 线索编号='LD-001'",
    "INSERT INTO 线索 (线索编号) VALUES ('x')",
    "DROP TABLE 线索",
    "ALTER TABLE 线索 ADD COLUMN x TEXT",
    "ATTACH DATABASE 'other.db' AS o",
    "PRAGMA table_info(线索)",
    "VACUUM",
])
def test_readonly_guard_rejects_write_statements(sql: str) -> None:
    with pytest.raises(ValueError, match="只读"):
        _assert_readonly(sql)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM 线索",
    "WITH t AS (SELECT * FROM 线索) SELECT * FROM t",
    "-- intermediate_sql\nSELECT COUNT(*) FROM 线索",
    "```sql\nSELECT * FROM 线索\n```",
    "/* 说明 */ SELECT * FROM 线索;",
])
def test_readonly_guard_allows_read_statements(sql: str) -> None:
    """前导注释/围栏必须先剥离，否则会把合法 SELECT 误杀（实测踩过一次）。"""
    _assert_readonly(sql)


def test_readonly_guard_rejects_comment_only_sql() -> None:
    with pytest.raises(ValueError, match="为空"):
        _assert_readonly("-- 只有注释没有语句")


# ── 受约束写入 ──────────────────────────────────────────────────────────
async def _write(**kwargs) -> str:
    return await SalesSqlWriteTool().execute(**kwargs)


@pytest.mark.asyncio
async def test_write_unique_hit_succeeds(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="线索编号", filter_value="LD-001",
                          target_column="负责人", new_value="赵磊")
    assert "成功" in result
    assert db.execute("SELECT 负责人 FROM 线索 WHERE 线索编号='LD-001'").fetchone()[0] == "赵磊"


@pytest.mark.asyncio
async def test_write_zero_hit_is_rejected(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="线索编号", filter_value="NOT-EXIST",
                          target_column="负责人", new_value="赵磊")
    assert "命中 0 行" in result and "未做任何修改" in result
    assert db.execute("SELECT COUNT(*) FROM 线索 WHERE 负责人='赵磊'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_write_multi_hit_is_rejected_with_count(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="行业", filter_value="制造业",
                          target_column="负责人", new_value="赵磊")
    assert "命中了 2 行" in result and "拒绝" in result
    assert db.execute("SELECT COUNT(*) FROM 线索 WHERE 负责人='赵磊'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_write_rejects_unknown_table(db: sqlite3.Connection) -> None:
    result = await _write(table="不存在的表", filter_column="线索编号", filter_value="LD-001",
                          target_column="负责人", new_value="x")
    assert "不存在" in result


@pytest.mark.asyncio
async def test_write_rejects_unknown_column_and_lists_real_ones(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="不存在的列", filter_value="LD-001",
                          target_column="负责人", new_value="x")
    assert "没有列" in result and "线索编号" in result


@pytest.mark.asyncio
async def test_write_parameterizes_values_so_injection_cannot_escape(db: sqlite3.Connection) -> None:
    """列名走白名单、值走 ? 占位：注入串既当不了列名，也改不动别的行。"""
    result = await _write(table="线索", filter_column="负责人' OR '1'='1", filter_value="x",
                          target_column="备注", new_value="y")
    assert "没有列" in result
    # 全表未被误改
    assert db.execute("SELECT COUNT(*) FROM 线索 WHERE 备注='y'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_write_rejects_incomplete_arguments(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="线索编号", target_column="负责人",
                          new_value="x")
    assert "必须提供" in result


# ── 初始化失败必须可见（不能静默不可用）────────────────────────────────
@pytest.mark.asyncio
async def test_query_reports_init_failure_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_BASE", "")
    reset_vanna()
    result = await SalesSqlQueryTool().execute(question="有多少条线索？")
    assert "初始化失败" in result


def test_get_vanna_raises_without_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_BASE", "")
    reset_vanna()
    with pytest.raises(RuntimeError, match="模型通道"):
        get_vanna()
    reset_vanna()


# ── LLM 通道必须显式超时/重试上限（修复"无限卡死"）──────────────────────
def test_sql_channel_defaults_bound_wait_time() -> None:
    """openai SDK 默认 600s×3 ≈ 30 分钟干等；钉死我们显式收敛后的默认值。"""
    assert 0 < sql_vanna_mod._SQL_LLM_TIMEOUT <= 120
    assert sql_vanna_mod._SQL_LLM_MAX_RETRIES <= 1


def test_vanna_client_receives_explicit_timeout_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """构造 SalesVanna 时必须把超时/重试显式传给 OpenAI 客户端，不能吃 SDK 默认。"""
    import openai

    captured: dict = {}

    class _Completions:
        def create(self, **kwargs):
            captured["create_kwargs"] = kwargs

            class _Msg:
                content = "SELECT 1"

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured["client_kwargs"] = kwargs
            self.chat = _Chat()

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    class _FakeChroma:
        def __init__(self, config=None) -> None:
            pass

        def connect_to_sqlite(self, path: str) -> None:
            pass

    class _FakeChat:
        def __init__(self, client=None, config=None) -> None:
            # 对齐真 OpenAI_Chat：submit_prompt 依赖这三个属性
            self.client = client
            self.config = config
            self.temperature = 0.7

    leaf_chroma = types.ModuleType("vanna.chromadb.chromadb_vector")
    leaf_chroma.ChromaDB_VectorStore = _FakeChroma
    leaf_chat = types.ModuleType("vanna.openai.openai_chat")
    leaf_chat.OpenAI_Chat = _FakeChat
    pkg_vanna = types.ModuleType("vanna")
    pkg_chroma = types.ModuleType("vanna.chromadb")
    pkg_chat = types.ModuleType("vanna.openai")
    pkg_chroma.chromadb_vector = leaf_chroma
    pkg_chat.openai_chat = leaf_chat
    for name, module in {
        "vanna": pkg_vanna,
        "vanna.chromadb": pkg_chroma,
        "vanna.chromadb.chromadb_vector": leaf_chroma,
        "vanna.openai": pkg_chat,
        "vanna.openai.openai_chat": leaf_chat,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.test/v1")
    sv = SalesVanna(db_path=str(tmp_path / "x.db"), chroma_path=str(tmp_path / "chroma"))

    assert captured["client_kwargs"]["api_key"] == "test-key"
    assert captured["client_kwargs"]["base_url"] == "https://example.test/v1"
    assert captured["client_kwargs"]["timeout"] == sql_vanna_mod._SQL_LLM_TIMEOUT
    assert captured["client_kwargs"]["max_retries"] == sql_vanna_mod._SQL_LLM_MAX_RETRIES

    # submit_prompt 默认必须关思考（qwen3 开思考实测 40~153s，关思考 3~16s）
    monkeypatch.setattr(sql_vanna_mod, "_SQL_LLM_ENABLE_THINKING", False)
    out = sv._vn.submit_prompt([{"role": "user", "content": "select 1"}])
    assert out == "SELECT 1"
    create_kwargs = captured["create_kwargs"]
    assert create_kwargs["extra_body"] == {"enable_thinking": False}
    assert create_kwargs["model"] == sql_vanna_mod._SQL_LLM_MODEL

    # 显式打开思考时不得注入 extra_body
    monkeypatch.setattr(sql_vanna_mod, "_SQL_LLM_ENABLE_THINKING", True)
    sv._vn.submit_prompt([{"role": "user", "content": "select 1"}])
    assert "extra_body" not in captured["create_kwargs"]


# ── 取数阻塞链路必须移出事件循环（修复"整页冻死"）──────────────────────
class _FakeVanna:
    def __init__(self, *, delay: float = 0.0, sql: str = "SELECT 1",
                 frame: object = None, exc: Exception | None = None) -> None:
        self._delay = delay
        self._sql = sql
        self._frame = frame
        self._exc = exc

    def generate_sql(self, question: str) -> str:
        time.sleep(self._delay)  # 复刻同步 HTTP 阻塞
        if self._exc is not None:
            raise self._exc
        return self._sql

    def run_readonly(self, sql: str) -> object:
        return self._frame


@pytest.mark.asyncio
async def test_execute_offloads_blocking_work_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """工具内部同步睡 1.2s 期间，事件循环上的并发心跳协程必须照常推进。

    旧实现直接在事件循环线程跑同步 HTTP：心跳一次都不会响（整页冻住）。
    """
    monkeypatch.setattr(sql_vanna_mod, "_vanna_singleton",
                        _FakeVanna(delay=1.2, sql=None))

    ticks: list[float] = []

    async def _heartbeat() -> None:
        deadline = asyncio.get_event_loop().time() + 1.0
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.1)
            ticks.append(asyncio.get_event_loop().time())

    tool_task = asyncio.create_task(SalesSqlQueryTool().execute(question="8月线索数"))
    heart_task = asyncio.create_task(_heartbeat())
    result = await tool_task
    await heart_task

    assert "未能生成有效的 SQL" in result
    assert len(ticks) >= 5, f"阻塞期间心跳只跳了 {len(ticks)} 次，事件循环被冻死"


@pytest.mark.asyncio
async def test_execute_returns_error_text_when_llm_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM 超时必须以错误 Observation 返回（Agent 可换问法），不能干等也不能抛穿。"""
    import openai

    fake_exc = openai.APITimeoutError(request=object())
    monkeypatch.setattr(sql_vanna_mod, "_vanna_singleton",
                        _FakeVanna(delay=0.0, exc=fake_exc))

    result = await SalesSqlQueryTool().execute(question="8月线索数")
    assert "SQL 生成或执行失败" in result
    assert "APITimeoutError" in result
