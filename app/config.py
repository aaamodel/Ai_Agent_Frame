# -*- coding: utf-8 -*-
"""应用配置：基于 pydantic-settings，支持环境变量与 .env 文件。

.env 搜索顺序（按优先级，后加载的同名变量不覆盖前面已有的）：
  1. 进程环境变量（`os.environ`，uvicorn 启动时自动注入或 CI/容器里设置）。
  2. `project-python/.env`（你现在写的那个，推荐放在仓库根目录）。
  3. `RAG-Challenge-2-main/.env`（旧路径，兜底保留，便于迁移）。

⚠️ .env 语法注意（python-dotenv 的硬性要求，不是 Pydantic）：
  - 注释 `# xxx` 必须独立一行，禁止行尾尾随（例：`FOO=bar # 注释` 是不合法的）。
  - 含有特殊字符（空格 / `[` `]` `{` `}` `,` `:` / `"` 等）的字符串请整体包双引号。
  - 不要用反引号 `` ` `` 代替引号（会被保留为字面字符，比如 `` `https://...` `` 会被当成字符串 `\`https://...\``）。
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# 辅助：字符串容错（去行尾注释、去外层引号、去反引号）
# ---------------------------------------------------------------------------
def _strip_trailing_comment(s: str) -> str:
    """把 `value # 中文注释` 这种 value 上的尾随注释去掉（不在 .env 解析层失败后也能兜底）。

    实现：只去掉『首个不在 "..." 或 '...' 内的 #』及之后的内容。
    """
    if not s:
        return s
    in_double = False
    in_single = False
    for idx, ch in enumerate(s):
        if ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "'" and not in_double:
            in_single = not in_single
        elif ch == "#" and not in_double and not in_single:
            # 注意：python-dotenv 里 # 前面要有空白才算注释；我们这里宽松一点
            if idx == 0 or s[idx - 1] in " \t":
                return s[:idx].rstrip()
    return s


def _unwrap_quotes_and_backticks(s: str) -> str:
    """去外层成对的 "" / '' / `` ``，并把反引号整体 URL 还原（`https://...` → https://...）。"""
    if not s:
        return s
    x = s
    # 循环剥掉一层：先 "" / ''，再 ``
    changed = True
    while changed and len(x) >= 2:
        changed = False
        pair = x[0] + x[-1]
        if pair in ('""', "''"):
            x = x[1:-1]
            changed = True
            continue
        if pair == "``":
            x = x[1:-1]
            changed = True
            continue
    return x.strip()


def _normalize_raw_value(raw: Any) -> Any:
    """在解析 list/dict 前，统一处理『字符串』的常见 .env 脏格式。"""
    if not isinstance(raw, str):
        return raw
    s = _strip_trailing_comment(raw)
    s = _unwrap_quotes_and_backticks(s)
    return s.strip()


# ---------------------------------------------------------------------------
# LLM_TIER_* / LLM_MODELS 解析器
# ---------------------------------------------------------------------------
def _parse_ids_from_str(raw: Any) -> List[str]:
    """将 str（JSON 数组 / 逗号分隔 / 空）标准化为 list[str]。"""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, tuple):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        s = _normalize_raw_value(raw)  # type: ignore[arg-type]
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass  # 回退逗号分隔
        return [item.strip() for item in s.split(",") if item.strip()]
    raise ValueError(f"无法解析为 id list: {raw!r}")


def _parse_llm_models_from_str(raw: Any) -> List[Dict[str, Any]]:
    """将 LLM_MODELS 字符串（JSON 数组）标准化为 list[dict]。

    额外容错：
      - 去尾随注释、去外层引号；
      - 对 python-dotenv 读到的『只等于 "[{" 半拉子 JSON』的情况，自动尝试从 os.environ 读原始字符串恢复；
      - 若 JSON 中含 `# 注释`（中文说明被夹进去了），也做一次字符串清理。
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return raw

    if not isinstance(raw, str):
        raise ValueError(f"llm_models 类型不合法：{type(raw)}")

    s = _normalize_raw_value(raw)  # type: ignore[arg-type]
    if not s:
        return []

    # 容错兜底：python-dotenv 有时会把未加引号的 JSON 截断（看到 {}/[] 特殊字符）
    # 这里直接从 os.environ 里再读一次 LLM_MODELS，优先用它（因为 uvicorn 启动期 load_dotenv 已经把完整的塞进 os.environ 了）
    env_raw = os.environ.get("LLM_MODELS") or ""
    if env_raw:
        env_clean = _normalize_raw_value(env_raw)
        # 谁更长谁更像"完整 JSON"（避免 parser 给的半截覆盖了真实的）
        if len(env_clean) > len(s):
            s = env_clean

    # 最后一次去外层的 ""
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()

    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM_MODELS 不是合法 JSON 数组：{e}. 原始片段={s[:120]!r}") from None
    if not isinstance(parsed, list):
        raise ValueError("LLM_MODELS 必须是 JSON 数组")
    # 逐项类型校验：dict，且 model_id 非空
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"LLM_MODELS[{idx}] 不是 JSON 对象: {item!r}")
        mid = str(item.get("model_id") or "").strip()
        if not mid:
            raise ValueError(f"LLM_MODELS[{idx}] 缺少 model_id 字段: {item!r}")
    return parsed


# ---------------------------------------------------------------------------
# Settings 本体
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """应用级配置。

    复杂字段（llm_models / llm_tier_*）在 Settings 里一律声明为 str，解析后通过 @property 暴露。
    原因：Pydantic Settings 的 EnvSettingsSource 会对 List[...] / Dict[...] 字段直接 json.loads(...)，
    对『逗号分隔』或『带行尾注释』这种常见 .env 写法容错很差。
    """

    model_config = SettingsConfigDict(
        env_file="./.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------- 应用基础 ----------------
    app_name: str = Field(default="Enterprise-aiagent", description="app 名称")
    app_env: str = Field(default="development", description="app 运行环境")
    debug: bool = Field(default=False, description="调试模式")
    api_prefix: str = Field(default="/api/v1", description="API 前缀")
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")

    # =========================================================================
    # 【单模型兼容入口】：只配这 3 项也能跑；3 tier 默认都复用 openai_llm_model。
    #   - .env 写法：  OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
    #                 （不要用反引号 `...`，不要行尾写 # 注释）
    # =========================================================================
    openai_api_key: str = Field(default="", description="OpenAI 兼容协议 API Key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI 兼容协议 BaseURL",
    )
    openai_llm_model: str = Field(default="deepseek-chat", description="默认对话模型（单模型模式）")

    # =========================================================================
    # 【多模型入口 · Tier 差异化路由】
    #
    # .env 示例（注意：注释必须独立一行，JSON 数组请整体用双引号包裹）：
    #   LLM_MODELS='[{"model_id":"glm-5.2","priority":0,"supports_thinking":true},{"model_id":"kimi-k2.7-code","priority":1,"supports_thinking":true}]'
    #   LLM_TIER_FAST=glm-5.2
    #   LLM_TIER_STANDARD=glm-5.2,kimi-k2.7-code
    #   LLM_TIER_DEEP=kimi-k2.7-code,glm-5.2
    #   LLM_TIER_FAST_TIMEOUT_MS=8000
    #   LLM_TIER_STANDARD_TIMEOUT_MS=25000
    #   LLM_TIER_DEEP_TIMEOUT_MS=90000
    # =========================================================================
    llm_models: str = Field(default="", description="多模型列表（JSON 数组字符串）")
    llm_tier_fast: str = Field(
        default="", description="FAST tier 候选 model_id 列表（逗号或 JSON 数组字符串，按顺序降级）"
    )
    llm_tier_standard: str = Field(
        default="", description="STANDARD tier 候选 model_id 列表"
    )
    llm_tier_deep: str = Field(
        default="", description="DEEP tier 候选 model_id 列表（至少 1 个 supports_thinking=true）"
    )
    # ⚠️ 超时语义：**单次尝试**的超时，不是"整个候选（含重试）"的总预算。
    #    最坏总耗时 = (1 + retries) × 本值，因此两者要一起看。
    llm_tier_fast_timeout_ms: int = Field(
        default=40_000, ge=1, description="FAST tier 单次尝试超时（ms）"
    )
    llm_tier_standard_timeout_ms: int = Field(
        default=60_000, ge=1, description="STANDARD tier 单次尝试超时（ms）"
    )
    llm_tier_deep_timeout_ms: int = Field(
        default=90_000, ge=1, description="DEEP tier 单次尝试超时（ms）"
    )
    # 重试由我们这一层显式控制（底层 SDK 的隐式重试已关闭），因此每次尝试各自
    # 享有完整的 timeout 预算；这里控制"额外尝试几次"（不含首次）。
    llm_tier_fast_retries: int = Field(
        default=1, ge=0, le=5, description="FAST tier 单次尝试失败后的重试次数（不含首次）"
    )
    llm_tier_standard_retries: int = Field(
        default=1, ge=0, le=5, description="STANDARD tier 单次尝试失败后的重试次数（不含首次）"
    )
    llm_tier_deep_retries: int = Field(
        default=1, ge=0, le=5, description="DEEP tier 单次尝试失败后的重试次数（不含首次）"
    )

    # ---------------- 后端 ----------------
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db",
        description="异步数据库 SQLAlchemy URL（postgresql+asyncpg）",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    milvus_host: str = Field(default="localhost", description="Milvus 主机")
    milvus_port: int = Field(default=19530, description="Milvus 端口")
    milvus_user: str = Field(default="", description="Milvus 用户名")
    milvus_password: str = Field(default="", description="Milvus 密码")
    milvus_collection_name: str = Field(
        default="agent_knowledge",
        description="默认向量名称",
    )
    milvus_kb_collection_name: str = Field(
        default="knowledge_base_v3",
        description="RAG 知识库集合名（LlamaIndex 重构后沿用旧集合名以便复用重建）",
    )

    # =========================================================================
    # Langfuse 可观测性（Agent 编排层观测调试）
    #   默认留空 → send_to_langfuse=False（不初始化、@observe 退化为 no-op）。
    #   配置 public_key + secret_key（+ 可选 host，默认 cloud.langfuse.com）后
    #   自动开启，观测点即可产生 Trace/Span/Generation 上报。
    # =========================================================================
    langfuse_public_key: str = Field(
        default="",
        description="Langfuse 公钥（pk-lf-...），为空则不启用观测",
    )
    langfuse_secret_key: str = Field(
        default="",
        description="Langfuse 私钥（sk-lf-...），为空则不启用观测",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse 服务地址（自建节点可替换）",
    )

    # =========================================================================
    # Agent 编排修复开关
    # =========================================================================
    enable_skill_tool_gating: bool = Field(
        default=True,
        description="（方案A）是否启用 Skill 号令工具：命中技能时，用技能 allowed-tools "
                    "与已注册工具取交集，覆盖 Pipeline 曾粗暴注入的工具集，避免非本技能工具污染候选。",
    )
    enable_empty_result_replan: bool = Field(
        default=True,
        description="（方案B）是否启用空业务结果触发重规划：数据源工具返回空数据（非异常）时，"
                    "停止执行剩余剧本并触发 replan，改用备选数据源，而非拿着空数据硬造结果。",
    )

    # =========================================================================
    # 2.2 状态图 / HITL 审批
    # =========================================================================
    react_max_steps: int = Field(
        default=10, ge=1,
        description="ReAct 态（无 plan execute 自环）单轮最大步数。",
    )
    max_replan_attempts: int = Field(
        # ⚠️ 由 2 改为 1（2026-09）：重规划的触发条件已收窄为"方向性错误"
        #（全部结论跑题），步级问题交由执行期就地纠偏消化。保留 1 次是给
        #"计划方向本身错了"这类情形留最后一条路——该情形只有重新规划能救。
        default=1, ge=0,
        description="plan 工具全坏/证据不足时最大重规划次数（收窄后至多 1 次）。",
    )
    agent_evidence_gate_enabled: bool = Field(
        default=True,
        description="summarize 证据充分性自判闸门：计划跑完后由汇总调用一并判定"
                    "证据是否足够作答，不足则带缺口说明 replan（0 额外 LLM 调用）。",
    )
    agent_checkpoint_backend: str = Field(
        default="redis",
        description="状态图 checkpointer 后端：redis（默认，失败自动降级内存）或 memory。",
    )
    agent_checkpoint_ttl_seconds: int = Field(
        default=86_400, ge=0,
        description="检查点 TTL（秒），0=永久；refresh_on_read 续期。",
    )
    agent_checkpoint_prefix: str = Field(
        default="agent_cp",
        description="Redis checkpoint key 前缀（多服务共库隔离）。",
    )
    agent_approval_enabled: bool = Field(
        default=True,
        description="危险工具人工审批总开关（HITL interrupt），默认开启。",
    )
    agent_danger_tools: str = Field(
        default="sales_sql_write,sales_report_export_tool",
        description="需人工审批的危险工具名，逗号分隔"
                    "（默认含业务库 SQL 写入与销售报表导出）。",
    )
    agent_reflect_enabled: bool = Field(
        default=False,
        description="reflect 质量门总开关，默认关闭（保持现网行为）。",
    )
    agent_reflect_min_score: int = Field(
        default=60, ge=0, le=100,
        description="reflect 质量通过分数线（0-100）。",
    )
    agent_node_retry_max: int = Field(
        default=1, ge=0,
        description="reflect 不通过时回 execute 重做的最大次数。",
    )

    log_level: str = Field(default="INFO", description="日志级别")

    # =========================================================================
    # 用户自定义 .env 基础 3 项 str 额外清理（反引号 URL / 尾随注释）
    # =========================================================================
    @model_validator(mode="before")
    @classmethod
    def _sanitize_scalar_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for key in ("openai_api_key", "openai_api_base", "openai_llm_model",
                    "llm_models", "llm_tier_fast", "llm_tier_standard", "llm_tier_deep"):
            if isinstance(data.get(key), str):
                cleaned = _strip_trailing_comment(data[key])
                cleaned = _unwrap_quotes_and_backticks(cleaned)
                data[key] = cleaned
        for key in ("llm_tier_fast_timeout_ms", "llm_tier_standard_timeout_ms",
                    "llm_tier_deep_timeout_ms",
                    "llm_tier_fast_retries", "llm_tier_standard_retries",
                    "llm_tier_deep_retries",
                    "milvus_port", "port"):
            if isinstance(data.get(key), str):
                cleaned = _strip_trailing_comment(data[key]).strip()
                data[key] = cleaned
        return data

    # ----- 在实例初始化后立刻把字符串解析成缓存属性（对外 *_parsed 属性访问）-----
    @model_validator(mode="after")
    def _parse_complex_fields(self) -> "Settings":
        object.__setattr__(
            self, "_parsed_llm_models", _parse_llm_models_from_str(self.llm_models)
        )
        object.__setattr__(
            self, "_parsed_tier_fast", _parse_ids_from_str(self.llm_tier_fast)
        )
        object.__setattr__(
            self, "_parsed_tier_standard", _parse_ids_from_str(self.llm_tier_standard)
        )
        object.__setattr__(
            self, "_parsed_tier_deep", _parse_ids_from_str(self.llm_tier_deep)
        )
        return self

    # 对外解析后属性：main.py / 业务代码只读这些
    @property
    def llm_models_parsed(self) -> List[Dict[str, Any]]:
        return getattr(
            self, "_parsed_llm_models", _parse_llm_models_from_str(self.llm_models)
        )

    @property
    def llm_tier_fast_parsed(self) -> List[str]:
        return getattr(
            self, "_parsed_tier_fast", _parse_ids_from_str(self.llm_tier_fast)
        )

    @property
    def llm_tier_standard_parsed(self) -> List[str]:
        return getattr(
            self, "_parsed_tier_standard", _parse_ids_from_str(self.llm_tier_standard)
        )

    @property
    def llm_tier_deep_parsed(self) -> List[str]:
        return getattr(
            self, "_parsed_tier_deep", _parse_ids_from_str(self.llm_tier_deep)
        )

    # 调试助手：把"解析结果"一次性 dump 成 dict（便于自检日志）
    def dump_llm(self) -> Dict[str, Any]:
        return {
            "env_file_resolved": self.model_config.get("env_file"),
            "openai_api_base": self.openai_api_base,
            "openai_llm_model": self.openai_llm_model,
            "llm_models": self.llm_models_parsed,
            "tiers": {
                "fast": {"candidates": self.llm_tier_fast_parsed,     "timeout_ms": self.llm_tier_fast_timeout_ms,     "retries": self.llm_tier_fast_retries},
                "standard": {"candidates": self.llm_tier_standard_parsed, "timeout_ms": self.llm_tier_standard_timeout_ms, "retries": self.llm_tier_standard_retries},
                "deep": {"candidates": self.llm_tier_deep_parsed,     "timeout_ms": self.llm_tier_deep_timeout_ms,     "retries": self.llm_tier_deep_retries},
            },
        }


@lru_cache
def get_settings() -> Settings:
    """单例：启动时初始化一次，进程内重复调用返回同一个对象（保证 LLM 解析结果一致）。"""
    return Settings()
