import { describe, expect, it } from "vitest";

import { parseSSEStream } from "./sse";

/** 把若干字符串分片喂给解析器，收集产出的事件。 */
async function collect(chunks: string[]) {
  const enc = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  const out: Record<string, unknown>[] = [];
  for await (const ev of parseSSEStream(stream)) out.push(ev);
  return out;
}

describe("parseSSEStream", () => {
  it("解析单条完整事件", async () => {
    const events = await collect(['data: {"content":"你好"}\n\n']);
    expect(events).toEqual([{ content: "你好" }]);
  });

  it("把跨 chunk 被切开的同一条事件拼回来", async () => {
    // ⚠️ Review Focus #1：TCP 分片会从任意位置切开，包括 JSON 中间
    const events = await collect([
      'data: {"cont',
      'ent":"被切开',
      '"}\n',
      "\n",
    ]);
    expect(events).toEqual([{ content: "被切开" }]);
  });

  it("一个 chunk 里塞多条事件时全部产出", async () => {
    const events = await collect([
      'data: {"content":"a"}\n\ndata: {"content":"b"}\n\ndata: {"done":true}\n\n',
    ]);
    expect(events).toEqual([{ content: "a" }, { content: "b" }, { done: true }]);
  });

  it("跳过解析不了的事件，但继续处理后续事件", async () => {
    // 不允许因一条坏事件中断整条流——否则用户看到回答莫名截断
    const events = await collect([
      "data: 这不是JSON\n\n",
      'data: {"content":"后续仍在"}\n\n',
    ]);
    expect(events).toEqual([{ content: "后续仍在" }]);
  });

  it("流结束时残留的半条事件被丢弃而不抛错", async () => {
    const events = await collect(['data: {"content":"未闭合']);
    expect(events).toEqual([]);
  });

  it("空 data 行不产出事件", async () => {
    const events = await collect(["\n\n", 'data: {"content":"x"}\n\n']);
    expect(events).toEqual([{ content: "x" }]);
  });
});
