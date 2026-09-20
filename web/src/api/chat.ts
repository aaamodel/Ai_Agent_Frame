import { apiDelete, apiPostJson, apiPostStream } from "@/lib/api";
import type { ChatMode } from "@/features/chat/types";

/** `DELETE /chat/sessions/{id}` 的响应：逐项状态，便于判断是否真的清干净。 */
export interface DeleteSessionResponse {
  session_id: string;
  /** 形如 {"short_term": "ok", "long_term": "failed: ..."} */
  memory: Record<string, string>;
  /** 顺带清掉的 Agent 检查点条数 */
  removed_runs: number;
}

/**
 * 删除一个会话（**含后端数据**）。
 *
 * ⚠️ 后端没有会话列表接口：消息正文存在前端 localStorage，后端另按 session_id
 * 存了短期记忆、长期记忆与 Agent 检查点。所以只删本地 = "删了还在"。
 */
export async function deleteChatSession(
  sessionId: string,
): Promise<DeleteSessionResponse> {
  return apiDelete<DeleteSessionResponse>(
    `/chat/sessions/${encodeURIComponent(sessionId)}`,
  );
}

/**
 * 发起 Agent 对话（工作任务模式），拿到 SSE 字节流。
 *
 * `strategy` 固定 `auto`（本设计不向用户暴露该参数）。
 */
export async function streamAgentChat(
  query: string,
  sessionId: string,
  signal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  return apiPostStream(
    "/chat/with_agent",
    { query, session_id: sessionId, strategy: "auto" },
    signal,
  );
}

/** `POST /chat` 的响应体（后端 `ChatResponse`，见 app/models/agent_schemas.py）。 */
export interface PlainChatResponse {
  /** 后端把 session_id 复用在这个字段里返回 */
  id: string;
  model: string;
  content: string;
  trace_id?: string | null;
  usage?: Record<string, unknown> | null;
}

/**
 * 闲聊模式：`POST /chat`。
 *
 * ⚠️ 与工作任务模式**返回形态不同**——这里不是 SSE，是一次性 JSON。
 * 请求体是 OpenAI 风格的 `messages` 数组（后端只取最后一条的 content），
 * 而不是 `/chat/with_agent` 的 `{query, session_id}`。
 */
export async function sendPlainChat(
  query: string,
  sessionId: string,
  signal?: AbortSignal,
): Promise<PlainChatResponse> {
  return apiPostJson<PlainChatResponse>(
    "/chat",
    {
      messages: [{ role: "user", content: query }],
      session_id: sessionId,
    },
    signal,
  );
}

/** 模式 → 实际调用的端点（集中在此，避免散落各处）。 */
export const MODE_ENDPOINT: Record<ChatMode, string> = {
  chat: "/chat",
  agent: "/chat/with_agent",
};
