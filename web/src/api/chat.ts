import { apiPostStream } from "@/lib/api";

/**
 * 发起 Agent 对话，拿到 SSE 字节流。
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
