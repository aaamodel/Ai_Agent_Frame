# -*- coding: utf-8 -*-
"""Locust 压测脚本：**真实 LLM** 端到端压测。

打的是应用真实端点（路径已核对 ``app/api/routes/chat.py`` + ``app/config.py::api_prefix``）：

=============================  ==============================  ==========================
端点                           请求体                          响应里有什么
=============================  ==============================  ==========================
``POST /api/v1/chat``          ``{messages, session_id,``      ``usage``（**真实 token**）
                               ``temperature, max_tokens}``
``POST /api/v1/chat/with_agent``  ``{query, session_id,``      SSE 流，**无 usage**
                                  ``strategy}``
=============================  ==============================  ==========================

> ⚠️ 手册里写的是 ``/v1/chat``，但项目实际前缀是 ``/api/v1``
> （``app/config.py:178  api_prefix = "/api/v1"``）。脚本按实际路径打。

## 为什么"单请求平均 token 成本"取 ``/chat`` 那一路

只有 ``/chat`` 的响应体带 ``usage``（见 ``chat.py:636  usage=resp.usage``），
Agent 那一路是 SSE 流、不带 usage。所以：
    - **token 成本**：来自 ``/chat`` 的响应（真实账单口径，非估算）；
    - **Agent 链路的 token**：需要从 Langfuse 取，见
      ``monitoring/langfuse_daily_report.py``。
报告里会把这两件事分开写，不混成一个数。

## 查询集

默认**复用 eval 的黄金集**（``evals/golden/intent_cases.jsonl`` +
``rag_cases.jsonl``）——这样压测的 query 分布与评测一致，
两边的数字可以互相解释。这是刻意的复用，不是偷懒。

## 用法

    # 50 并发跑 10 分钟，出 CSV
    BENCH_HOST=http://127.0.0.1:8000 \
      locust -f benchmark/locustfile.py --headless -u 50 -r 5 -t 10m \
             --csv benchmark/_results/locust_50 --only-summary

    # 或直接用编排脚本
    bash benchmark/run_bench.sh 50 10m
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from locust import HttpUser, between, events, task

_BENCH_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _BENCH_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RESULTS_DIR: Path = _BENCH_DIR / "_results"

# ---------------------------------------------------------------------
# 端点（可用环境变量覆盖；默认值已按源码核实）
# ---------------------------------------------------------------------
CHAT_ENDPOINT: str = os.getenv("BENCH_CHAT_ENDPOINT", "/api/v1/chat")
AGENT_ENDPOINT: str = os.getenv("BENCH_AGENT_ENDPOINT", "/api/v1/chat/with_agent")
SESSION_PREFIX: str = os.getenv("BENCH_SESSION_PREFIX", "bench")

# 任务权重：整数比。默认 Agent 重一点，因为它是主链路
WEIGHT_CHAT: int = int(os.getenv("BENCH_WEIGHT_CHAT", "1"))
WEIGHT_AGENT: int = int(os.getenv("BENCH_WEIGHT_AGENT", "2"))

# 请求超时：Agent 链路多步 LLM，给足时间（否则会误记为失败）
REQUEST_TIMEOUT_S: float = float(os.getenv("BENCH_TIMEOUT_S", "180"))


# =====================================================================
# 查询集：默认复用 eval 黄金集
# =====================================================================
def _load_queries_from_golden() -> List[str]:
    queries: List[str] = []
    golden_dir: Path = _REPO_ROOT / "evals" / "golden"
    for name in ("intent_cases.jsonl", "rag_cases.jsonl", "tool_cases.jsonl"):
        path: Path = golden_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text: str = line.strip()
                if not text or text.startswith("#"):
                    continue
                try:
                    record: Any = json.loads(text)
                except json.JSONDecodeError:
                    continue
                query: Any = record.get("query") if isinstance(record, dict) else None
                if isinstance(query, str) and query.strip():
                    queries.append(query.strip())
    return queries


def _load_queries() -> List[str]:
    """优先读 BENCH_QUERIES 指定的文件（每行一条），否则复用黄金集。"""
    custom: str = os.getenv("BENCH_QUERIES", "").strip()
    if custom:
        path: Path = Path(custom)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                # 跳过空行与 # 注释行：问题集文件需要能自带选取说明
                queries: List[str] = [
                    ln.strip()
                    for ln in handle
                    if ln.strip() and not ln.strip().startswith("#")
                ]
            if queries:
                print(f"[bench] 从 {path} 载入 {len(queries)} 条查询")
                return queries
        print(f"[bench] ⚠️ BENCH_QUERIES={custom} 不存在，回退到黄金集")
    queries = _load_queries_from_golden()
    if not queries:  # 极端兜底：黄金集也没有
        queries = ["智能客服平台专业版报价是多少？", "客户说竞品更便宜，怎么回应？"]
    print(f"[bench] 使用黄金集查询 {len(queries)} 条（与 eval 同分布）")
    return queries


QUERIES: List[str] = _load_queries()


# =====================================================================
# token 用量累计（线程安全；压测结束写盘）
# =====================================================================
TOKEN_LOCK = threading.Lock()
TOKEN_STATS: Dict[str, Any] = {
    "chat_requests_with_usage": 0,
    "chat_requests_without_usage": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "by_model": {},
    "agent_requests": 0,
    "agent_requests_failed": 0,
}


def _record_usage(model: str, usage: Any) -> None:
    if not isinstance(usage, dict):
        with TOKEN_LOCK:
            TOKEN_STATS["chat_requests_without_usage"] += 1
        return
    prompt: int = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion: int = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    total: int = int(usage.get("total_tokens") or (prompt + completion))
    with TOKEN_LOCK:
        TOKEN_STATS["chat_requests_with_usage"] += 1
        TOKEN_STATS["prompt_tokens"] += prompt
        TOKEN_STATS["completion_tokens"] += completion
        TOKEN_STATS["total_tokens"] += total
        bucket: Dict[str, Any] = TOKEN_STATS["by_model"].setdefault(
            model or "unknown", {"requests": 0, "input": 0, "output": 0, "total": 0}
        )
        bucket["requests"] += 1
        bucket["input"] += prompt
        bucket["output"] += completion
        bucket["total"] += total


@events.quitting.add_listener
def _dump_stats(environment: Any, **_kwargs: Any) -> None:
    """压测结束时把 token 统计落盘，供报告引用。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with TOKEN_LOCK:
        snapshot: Dict[str, Any] = json.loads(json.dumps(TOKEN_STATS))
    snapshot["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    snapshot["chat_endpoint"] = CHAT_ENDPOINT
    snapshot["agent_endpoint"] = AGENT_ENDPOINT
    snapshot["query_count"] = len(QUERIES)
    snapshot["weights"] = {"chat": WEIGHT_CHAT, "agent": WEIGHT_AGENT}

    out: Path = RESULTS_DIR / "token_usage.json"
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    with_usage: int = snapshot["chat_requests_with_usage"]
    avg: str = (
        f"{snapshot['total_tokens'] / with_usage:.0f}"
        if with_usage
        else "不可用（响应中没有 usage）"
    )
    print(f"\n[bench] token 统计 -> {out}")
    print(f"[bench]   带 usage 的 /chat 请求：{with_usage}")
    print(f"[bench]   单请求平均 token：{avg}")


# =====================================================================
# 用户行为
# =====================================================================
class ChatUser(HttpUser):
    """混合压测用户：标准对话 + Agent 链路。

    ``wait_time`` 用 0.5~2s 而不是 0：真实用户会读回复再提问；
    如果打满不停，测出来的是"连接池极限"而不是"业务容量"，
    和手册要的 QPS 不是一个东西。
    """

    # host 由 --host 提供；这里给个默认值方便本地直接跑
    host: str = "http://127.0.0.1:8000"
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        # 每个虚拟用户一个独立 session，避免会话记忆互相污染
        self.session_id: str = f"{SESSION_PREFIX}-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    def _next_query(self) -> str:
        return random.choice(QUERIES)

    # ------------------------------------------------------------------
    @task(WEIGHT_CHAT)
    def chat_nonstream(self) -> None:
        """``POST /api/v1/chat``：响应带真实 usage，是 token 成本的数据来源。"""
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": self._next_query()}],
            "session_id": self.session_id,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        with self.client.post(
            CHAT_ENDPOINT,
            json=payload,
            name=f"POST {CHAT_ENDPOINT}",
            timeout=REQUEST_TIMEOUT_S,
            catch_response=True,
        ) as response:
            if response.status_code >= 400:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                body: Any = response.json()
            except ValueError:
                response.failure("响应不是 JSON")
                return
            if not isinstance(body, dict):
                response.failure("响应不是对象")
                return
            # 空内容也算业务失败：它对用户就是"没回答"
            if not str(body.get("content") or "").strip():
                response.failure("content 为空（业务失败）")
            _record_usage(str(body.get("model") or "unknown"), body.get("usage"))

    # ------------------------------------------------------------------
    @task(WEIGHT_AGENT)
    def agent_chat(self) -> None:
        """``POST /api/v1/chat/with_agent``：SSE 主链路。

        说明：这里按"读完整条流"来计时，得到的是**端到端完成时间**。
        首字延迟（TTFT）不单独测量——需要与前端一致地解析 SSE 事件边界，
        属于已知未覆盖项，已在报告里写明。
        """
        payload: Dict[str, Any] = {
            "query": self._next_query(),
            "session_id": self.session_id,
            "strategy": os.getenv("BENCH_STRATEGY", "auto"),
        }
        with TOKEN_LOCK:
            TOKEN_STATS["agent_requests"] += 1
        with self.client.post(
            AGENT_ENDPOINT,
            json=payload,
            name=f"POST {AGENT_ENDPOINT}",
            timeout=REQUEST_TIMEOUT_S,
            catch_response=True,
        ) as response:
            if response.status_code >= 400:
                with TOKEN_LOCK:
                    TOKEN_STATS["agent_requests_failed"] += 1
                response.failure(f"HTTP {response.status_code}")
                return
            text: str = response.text or ""
            # 空流同样是业务失败
            if not text.strip():
                with TOKEN_LOCK:
                    TOKEN_STATS["agent_requests_failed"] += 1
                response.failure("SSE 流为空（业务失败）")

    # ------------------------------------------------------------------
    @task(0)
    def health(self) -> None:
        """权重 0：默认不跑；需要单独验证服务可达时把权重调大即可。"""
        self.client.get("/api/v1/health", name="GET /api/v1/health")
