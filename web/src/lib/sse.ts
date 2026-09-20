/**
 * 解析本项目后端自定义的 SSE 流。
 *
 * 后端格式（app/api/routes/chat.py 的 `_sse_payload`）：
 *     data: {"content":"…"}\n\n
 *
 * ⚠️ 不能用 `EventSource`：它是 GET-only，而 /chat/with_agent 是 POST。
 * 因此这里手写解析，必须自己处理**事件跨 chunk 被切开**的情况。
 */
export async function* parseSSEStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<Record<string, unknown>> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      // stream:true 保证多字节 UTF-8 字符被切开时也能正确拼回
      buffer += decoder.decode(value, { stream: true });

      // 事件以空行分隔。注意用 \n\n，且要处理 \r\n\n 这类变体。
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const parsed = parseOneEvent(rawEvent);
        if (parsed) yield parsed;
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
  // 结束时 buffer 里残留的半条事件**有意丢弃**：
  // 它意味着连接在事件中途断了，半条 JSON 无法可靠复原。
}

function parseOneEvent(rawEvent: string): Record<string, unknown> | null {
  const dataLines = rawEvent
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());

  if (dataLines.length === 0) return null;
  const payload = dataLines.join("\n").trim();
  if (!payload) return null;

  try {
    const parsed: unknown = JSON.parse(payload);
    // 只接受对象——数组/字符串/数字都不是本后端的合法事件
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    // 坏事件跳过而非抛出：一条解析不了的事件不该让整条回答消失
    return null;
  }
}
