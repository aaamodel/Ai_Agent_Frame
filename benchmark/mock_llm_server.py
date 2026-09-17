# -*- coding: utf-8 -*-
"""Mock LLM 服务：OpenAI 兼容协议，用来**零 API 成本**压测框架自身开销。

## 为什么需要它

真实压测的延迟里，**LLM 推理占了 80%~95%**，框架自身（路由/意图/检索/编排）
的开销被淹没在噪声里。而且真实压测烧钱、有限流、结果还不可复现。

所以压测分两轮：

=========================  ====================  ==========================
轮次                        LLM 来源              能回答什么问题
=========================  ====================  ==========================
第一轮（本文件）            mock server           框架自身能扛多少 QPS？
                                                 连接池/事件循环/序列化是不是瓶颈？
第二轮（locustfile.py）     真实模型 API          端到端 P95 / 单轮成本是多少？
=========================  ====================  ==========================

## 为什么能完全零成本

应用的两个外部依赖都走 **OpenAI 兼容协议**，把 ``OPENAI_API_BASE`` 指向本服务即可：

- 对话：``app/llm_model_router/async_openai_caller.py`` → ``AsyncOpenAI(base_url=...)``
- 向量：``app/infrastructure/embeddings/dashscope_embedding.py`` → ``OpenAI(base_url=...)``

于是**连 embedding 都不用调真实 API**，Milvus 检索全链路都能在 mock 下跑起来。
（注意：Milvus 本身仍需真实运行；它不在 mock 范围内。）

只依赖 Python 标准库（``http.server``），不需要 FastAPI/uvicorn。

## 用法

    # 默认：监听 8100，模拟 300ms 推理延迟
    python benchmark/mock_llm_server.py

    # 模拟"主模型挂了"：名为 qwen3.8-max 的请求直接超时（测熔断降级）
    python benchmark/mock_llm_server.py --hang-model qwen3.8-max --hang-seconds 60

    # 模拟随机故障（测错误率统计）
    python benchmark/mock_llm_server.py --fail-rate 0.05

    # 查看统计（mock 侧真实收到的调用数与字符数）
    curl http://127.0.0.1:8100/stats

## token 口径（重要，别把 mock 数字当真实成本）

mock 不知道真实分词器，因此 ``usage`` 由下面这个**明确记录的估算式**给出：

    tokens ≈ 0.75 × CJK 字符数 + 0.25 × 非 CJK 字符数

它只用来验证"token 字段有没有被链路正确透传"，**不能当作成本结论**。
响应里同时带 ``mock_estimated: true`` 和原始字符数，避免被误读。
真实成本请走 ``monitoring/langfuse_daily_report.py``（读 Langfuse 的真实 usage）。
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_PORT: int = 8100
DEFAULT_DIM: int = 1024

# ---------------------------------------------------------------------
# 运行时配置（由 CLI 写入，handler 读取）
# ---------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "latency_ms": 300.0,
    "jitter_ms": 100.0,
    "fail_rate": 0.0,
    "hang_model": None,
    "hang_seconds": 60.0,
    "fail_model": None,
    "tool_call_ratio": 0.0,
    "verbose": False,
}

STATS_LOCK = threading.Lock()
STATS: Dict[str, Any] = {
    "chat_requests": 0,
    "embedding_requests": 0,
    "failed_requests": 0,
    "hanged_requests": 0,
    "prompt_chars": 0,
    "completion_chars": 0,
    "by_model": {},
    "started_at": time.time(),
}


def _bump(field: str, delta: int = 1) -> None:
    with STATS_LOCK:
        STATS[field] = STATS.get(field, 0) + delta


def _bump_model(model: str, field: str, delta: int) -> None:
    with STATS_LOCK:
        bucket: Dict[str, Any] = STATS["by_model"].setdefault(
            model or "unknown", {"chat": 0, "embedding": 0, "errors": 0}
        )
        bucket[field] = bucket.get(field, 0) + delta


# ---------------------------------------------------------------------
# 估算 token（口径见模块 docstring，必须保持"可复现、不假装真实"）
# ---------------------------------------------------------------------
def approx_tokens(text: str) -> int:
    """CJK 0.75 token/字、非 CJK 0.25 token/字符的估算。"""
    if not text:
        return 0
    cjk: int = 0
    other: int = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff":
            cjk += 1
        else:
            other += 1
    return max(1, int(cjk * 0.75 + other * 0.25))


def _messages_text(messages: Any) -> str:
    """把 messages 拍平成纯文本（用于估算 prompt token 数与长度分布）。"""
    if not isinstance(messages, list):
        return str(messages or "")
    parts: List[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
    return "\n".join(parts)


def _fake_embedding(text: str, dim: int) -> List[float]:
    """确定性伪向量：同一文本永远得到同一向量。

    为什么不用随机向量：BM25 之外还有向量通道，如果每次查询得到不同向量，
    检索结果会漂移，压测结果就不可复现。这里用字符哈希做种子，保证可复现。
    向量做了 L2 归一化（与真实 embedding 的 IP 度量习惯一致）。
    """
    seed: int = 0
    for ch in text:
        seed = (seed * 131 + ord(ch)) & 0x7FFFFFFF
    rnd = random.Random(seed)
    vec: List[float] = [rnd.uniform(-1.0, 1.0) for _ in range(dim)]
    norm: float = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


# ---------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------
class MockLLMHandler(BaseHTTPRequestHandler):
    """OpenAI 兼容的极简实现。"""

    protocol_version = "HTTP/1.1"
    server_version = "MockLLM/1.0"

    # ---- 工具 ----
    def _send_json(self, payload: Any, status: int = 200) -> None:
        body: bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 500) -> None:
        self._send_json(
            {"error": {"message": message, "type": "mock_error", "code": status}},
            status=status,
        )

    def _read_body(self) -> Dict[str, Any]:
        length: int = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw: bytes = self.rfile.read(length)
        try:
            data: Any = json.loads(raw.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _simulate_latency(self) -> None:
        latency: float = float(CONFIG["latency_ms"]) + random.uniform(
            0.0, float(CONFIG["jitter_ms"])
        )
        if latency > 0:
            time.sleep(latency / 1000.0)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - 基类签名
        if CONFIG["verbose"]:
            super().log_message(fmt, *args)

    # ---- keep-alive 下的客户端断开兜底 ----
    def handle_one_request(self) -> None:
        """吞掉「客户端主动断开」导致的 ConnectionResetError。

        为什么必须处理：``protocol_version = "HTTP/1.1"`` 会开启 keep-alive，
        压测工具（locust/curl）在收到响应后常常直接关连接。此时 ``rfile.readline``
        会抛 ``ConnectionResetError``，``socketserver`` 会把它当成「请求处理异常」
        打一整屏 traceback——**看起来像 mock 崩了，其实只是正常断开**。
        把连接标记为关闭即可，不改变任何响应语义。
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            self.close_connection = True

    # ---- 路由 ----
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        if self.path.startswith("/healthz"):
            self._send_json({"status": "ok"})
            return
        if self.path.startswith("/stats"):
            with STATS_LOCK:
                snapshot: Dict[str, Any] = json.loads(json.dumps(STATS))
            snapshot["uptime_seconds"] = round(time.time() - STATS["started_at"], 1)
            snapshot["config"] = CONFIG
            self._send_json(snapshot)
            return
        if self.path.startswith("/admin/reset"):
            with STATS_LOCK:
                for key in list(STATS):
                    if key not in {"started_at"}:
                        STATS[key] = {} if key == "by_model" else 0
            self._send_json({"status": "reset"})
            return
        self._send_error_json(f"未知路径 {self.path}", status=404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        if self.path.endswith("/chat/completions"):
            self._handle_chat()
            return
        if self.path.endswith("/embeddings"):
            self._handle_embeddings()
            return
        self._send_error_json(f"未知路径 {self.path}", status=404)

    # ---- /v1/chat/completions ----
    def _handle_chat(self) -> None:
        body: Dict[str, Any] = self._read_body()
        model: str = str(body.get("model") or "mock-model")
        _bump("chat_requests")
        _bump_model(model, "chat", 1)

        # 故障注入 1：指定模型挂起（测熔断需要"超时"而不是"立刻报错"）
        if CONFIG["hang_model"] and model == CONFIG["hang_model"]:
            _bump("hanged_requests")
            _bump_model(model, "errors", 1)
            time.sleep(float(CONFIG["hang_seconds"]))
            self._send_error_json("mock: 模型超时未响应", status=504)
            return

        # 故障注入 2：指定模型立即报错
        if CONFIG["fail_model"] and model == CONFIG["fail_model"]:
            _bump("failed_requests")
            _bump_model(model, "errors", 1)
            self._send_error_json(f"mock: 模型 {model} 不可用", status=500)
            return

        # 故障注入 3：按比例随机失败（测错误率统计）
        if CONFIG["fail_rate"] and random.random() < float(CONFIG["fail_rate"]):
            _bump("failed_requests")
            _bump_model(model, "errors", 1)
            self._send_error_json("mock: 随机故障注入", status=503)
            return

        self._simulate_latency()

        prompt_text: str = _messages_text(body.get("messages"))
        prompt_tokens: int = approx_tokens(prompt_text)
        _bump("prompt_chars", len(prompt_text))

        tool_calls: Optional[List[Dict[str, Any]]] = None
        tools: Any = body.get("tools")
        if tools and random.random() < float(CONFIG["tool_call_ratio"]):
            # 返回一个工具调用，用于把 Agent 的工具路径也压进来
            first_name: str = ""
            if isinstance(tools, list) and tools:
                fn: Any = (tools[0] or {}).get("function") or {}
                first_name = str(fn.get("name") or "")
            if first_name:
                tool_calls = [
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": first_name,
                            "arguments": json.dumps(
                                {"query": prompt_text[:60] or "mock"}, ensure_ascii=False
                            ),
                        },
                    }
                ]

        content: str = "" if tool_calls else "这是 mock LLM 的合成回答，用于压测框架自身开销。"
        completion_tokens: int = approx_tokens(content)
        _bump("completion_chars", len(content))

        message: Dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls

        self._send_json(
            {
                "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    # 这两个字段是刻意加的：防止把 mock 估算值当成真实账单口径
                    "mock_estimated": True,
                    "estimator": "0.75*cjk_chars+0.25*other_chars",
                },
            }
        )

    # ---- /v1/embeddings ----
    def _handle_embeddings(self) -> None:
        body: Dict[str, Any] = self._read_body()
        model: str = str(body.get("model") or "mock-embedding")
        raw_input: Any = body.get("input")
        texts: List[str] = (
            [str(t) for t in raw_input]
            if isinstance(raw_input, list)
            else [str(raw_input or "")]
        )
        _bump("embedding_requests")
        _bump_model(model, "embedding", 1)

        if CONFIG["fail_rate"] and random.random() < float(CONFIG["fail_rate"]):
            _bump("failed_requests")
            self._send_error_json("mock: embedding 随机故障", status=503)
            return

        # embedding 不注入推理延迟：它的真实耗时由 Milvus/网络决定，
        # 这里只负责"便宜地提供向量"，避免把 mock 的 sleep 混进检索耗时。
        data: List[Dict[str, Any]] = [
            {"object": "embedding", "index": i, "embedding": _fake_embedding(t, DEFAULT_DIM)}
            for i, t in enumerate(texts)
        ]
        total_chars: int = sum(len(t) for t in texts)
        _bump("prompt_chars", total_chars)

        self._send_json(
            {
                "object": "list",
                "data": data,
                "model": model,
                "usage": {
                    "prompt_tokens": approx_tokens("".join(texts)),
                    "total_tokens": approx_tokens("".join(texts)),
                    "mock_estimated": True,
                },
            }
        )


def serve(
    host: str,
    port: int,
    *,
    latency_ms: float = 300.0,
    jitter_ms: float = 100.0,
    fail_rate: float = 0.0,
    hang_model: Optional[str] = None,
    hang_seconds: float = 60.0,
    fail_model: Optional[str] = None,
    tool_call_ratio: float = 0.0,
    verbose: bool = False,
) -> None:
    """启动 mock 服务（阻塞）。"""
    CONFIG.update(
        {
            "latency_ms": latency_ms,
            "jitter_ms": jitter_ms,
            "fail_rate": fail_rate,
            "hang_model": hang_model,
            "hang_seconds": hang_seconds,
            "fail_model": fail_model,
            "tool_call_ratio": tool_call_ratio,
            "verbose": verbose,
        }
    )
    server = ThreadingHTTPServer((host, port), MockLLMHandler)
    server.daemon_threads = True
    print(
        f"[mock-llm] 监听 http://{host}:{port}\n"
        f"[mock-llm]   POST /v1/chat/completions  (latency={latency_ms}ms"
        f"+{jitter_ms}ms jitter, fail_rate={fail_rate})\n"
        f"[mock-llm]   POST /v1/embeddings        (确定性伪向量, dim={DEFAULT_DIM})\n"
        f"[mock-llm]   GET  /stats | /healthz | /admin/reset\n"
        f"[mock-llm] 把应用指向它: OPENAI_API_BASE=http://{host}:{port}/v1\n"
        f"[mock-llm] ⚠️ usage 中的 token 为估算值（mock_estimated=true），"
        f"不可用作成本结论",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock-llm] 收到中断，退出。")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容的 mock LLM 服务（零 API 成本压测用）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--latency-ms", type=float, default=300.0, help="模拟推理延迟")
    parser.add_argument("--jitter-ms", type=float, default=100.0, help="延迟抖动上限")
    parser.add_argument("--fail-rate", type=float, default=0.0, help="随机失败比例 0~1")
    parser.add_argument("--hang-model", default=None, help="该模型名请求将被挂起（测熔断）")
    parser.add_argument("--hang-seconds", type=float, default=60.0, help="挂起时长")
    parser.add_argument("--fail-model", default=None, help="该模型名请求立即报错")
    parser.add_argument(
        "--tool-call-ratio",
        type=float,
        default=0.0,
        help="请求带 tools 时返回 tool_call 的比例（把工具路径也压进来）",
    )
    parser.add_argument("--verbose", action="store_true", help="打印每个请求日志")
    args = parser.parse_args()

    serve(
        args.host,
        args.port,
        latency_ms=args.latency_ms,
        jitter_ms=args.jitter_ms,
        fail_rate=args.fail_rate,
        hang_model=args.hang_model,
        hang_seconds=args.hang_seconds,
        fail_model=args.fail_model,
        tool_call_ratio=args.tool_call_ratio,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
