# -*- coding: utf-8 -*-
"""Locust 压测脚本：**mock LLM** 模式，专测框架自身开销（零 API 成本）。

和 ``locustfile.py`` 的唯一区别是**被测系统的后端**：应用被配置成
``OPENAI_API_BASE=http://127.0.0.1:8100/v1``，于是对话与 embedding 都打到
mock 服务上。这样测出来的延迟/吞吐**剔除了模型推理时间与外部 API 抖动**，
剩下的就是框架自身的开销（路由/意图三阶段/混合检索/编排/序列化/连接池）。

## 本脚本额外做两件 locustfile.py 不做的事

1. **验证 mock 模式确实生效**（``MockProbeUser``）：直接打 mock 的 ``/stats``，
   确认 mock 侧的 ``chat_requests`` 在增长。否则可能出现"你以为在测 mock，
   其实应用还在打真实 API"——那样既烧钱又得不出干净结论。
2. **测 mock 自身的吞吐上限**（``MockOnlyUser``）：直接压 mock server。
   如果 mock 自己就撑不住，那么应用那边的 QPS 数字就是**mock 的瓶颈**
   而不是应用的瓶颈——这个对照能避免得出错误结论。

用法::

    # 先起 mock
    python benchmark/mock_llm_server.py --latency-ms 300 &

    # 再起应用（指向 mock）
    OPENAI_API_BASE=http://127.0.0.1:8100/v1 uvicorn app.main:app --port 8000 &

    # 压测
    BENCH_HOST=http://127.0.0.1:8000 locust -f benchmark/locustfile_mock.py \
      --headless -u 50 -r 5 -t 10m --csv benchmark/_results/mock_50 --only-summary
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from locust import HttpUser, between, task

_BENCH_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _BENCH_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

# 复用主压测脚本的配置与用户行为，避免两份实现漂移
from locustfile import (  # noqa: E402
    AGENT_ENDPOINT,
    CHAT_ENDPOINT,
    QUERIES,
    TOKEN_STATS,
    ChatUser,
)

MOCK_BASE: str = os.getenv("BENCH_MOCK_BASE", "http://127.0.0.1:8100")

__all__ = ["MockBackedChatUser", "MockProbeUser", "MockOnlyUser", "ChatUser"]


class MockBackedChatUser(ChatUser):
    """与应用主链路完全相同的用户行为，只是后端跑在 mock 上。

    直接继承 ``ChatUser``：确保「mock 组」与「真实组」打的是同一套请求体、
    同一套查询分布、同一套权重。两组数字才有可比性——这是 A/B 的前提。
    """

    # 给一个 MockOnlyUser 触发 mock 自检的钩子
    def on_start(self) -> None:
        super().on_start()
        # 自检：确认 mock 在跑（否则这轮数字没有意义）
        try:
            resp = self.client.get(
                f"{MOCK_BASE}/healthz", name="MOCK GET /healthz", timeout=5
            )
            if resp.status_code >= 400:
                print(
                    f"[mock-bench] ⚠️ mock 服务不可达（HTTP {resp.status_code}）。"
                    f"请先运行：python benchmark/mock_llm_server.py"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[mock-bench] ⚠️ mock 服务不可达：{exc}")


class MockProbeUser(HttpUser):
    """低频探针：周期性核对 mock 侧的调用计数在增长。

    作用：证明"应用确实打到了 mock"，而不是静默打到了真实 API。
    权重由 ``BENCH_PROBE_USERS`` 控制（默认 1 个用户），不参与主压测统计口径。
    """

    host: str = MOCK_BASE
    wait_time = between(5.0, 10.0)

    @task
    def probe_stats(self) -> None:
        with self.client.get(
            f"{MOCK_BASE}/stats", name="MOCK GET /stats", timeout=10, catch_response=True
        ) as response:
            if response.status_code >= 400:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                body: Dict[str, Any] = response.json()
            except ValueError:
                response.failure("stats 不是 JSON")
                return
            chat_requests: int = int(body.get("chat_requests") or 0)
            embedding_requests: int = int(body.get("embedding_requests") or 0)
            print(
                f"[mock-bench] mock 侧计数：chat={chat_requests} "
                f"embedding={embedding_requests}"
            )
            if chat_requests == 0 and embedding_requests == 0:
                response.failure(
                    "mock 计数为 0：应用可能没有打到 mock（检查应用的 OPENAI_API_BASE）"
                )


class MockOnlyUser(HttpUser):
    """直接压 mock server：用来测出 mock 自身的吞吐上限作为对照基准。

    为什么需要：如果 mock 自己 P95 就已经很高，那么"应用 QPS 上不去"
    只能说明**mock 是瓶颈**，不能归因于框架。这个对照组就是为了避免
    把 mock 的锅算到框架头上。
    """

    host: str = MOCK_BASE
    wait_time = between(0.1, 0.5)

    @task
    def chat_completions(self) -> None:
        payload: Dict[str, Any] = {
            "model": os.getenv("BENCH_MOCK_MODEL", "mock-model"),
            "messages": [{"role": "user", "content": "压测用的合成问题"}],
            "temperature": 0.7,
        }
        self.client.post(
            f"{MOCK_BASE}/v1/chat/completions",
            json=payload,
            name="MOCK POST /v1/chat/completions",
            timeout=60,
        )

    @task
    def embeddings(self) -> None:
        payload: Dict[str, Any] = {
            "model": "text-embedding-v3",
            "input": ["混合检索压测用的合成文本"],
        }
        self.client.post(
            f"{MOCK_BASE}/v1/embeddings",
            json=payload,
            name="MOCK POST /v1/embeddings",
            timeout=60,
        )
