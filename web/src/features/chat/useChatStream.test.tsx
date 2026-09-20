import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as approvalsApi from "@/api/approvals";
import * as chatApi from "@/api/chat";

import type { ChatMessage } from "./types";
import { useChatStream } from "./useChatStream";

/** 造一个按顺序吐 SSE 文本的假流。 */
function sseStream(chunks: string[]) {
  const enc = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(c) {
      for (const s of chunks) c.enqueue(enc.encode(s));
      c.close();
    },
  });
}

/**
 * 把消息更新收集进一个可变数组，模拟页面侧的消息列表。
 *
 * ⚠️ 首次更新前必须**先压入一条助手占位消息** —— 这正是真实 ChatPage 的行为：
 * 用户发问时先追加一条空的助手消息，再把流式内容往里写。
 * 少了这一步，`messages.length - 1` 会是 -1，`updater(undefined)` 直接抛错。
 */
function collect() {
  const messages: ChatMessage[] = [];
  const onMessage = (updater: (m: ChatMessage) => ChatMessage) => {
    if (messages.length === 0) {
      messages.push({ id: "a1", role: "assistant", text: "" });
    }
    const idx = messages.length - 1;
    messages[idx] = updater(messages[idx]);
  };
  return { messages, onMessage };
}

afterEach(() => vi.restoreAllMocks());

describe("useChatStream", () => {
  it("累积正文片段并在 done 时写入元数据", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"content":"优先行业"}\n\n',
        'data: {"content":"为金融"}\n\n',
        'data: {"done":true,"status":"success","steps_executed":3,"trace_id":"t1","session_id":"s1"}\n\n',
      ]),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));

    await act(async () => {
      await result.current.send("问题", "s1");
    });

    expect(messages[0]?.text).toBe("优先行业为金融");
    expect(messages[0]?.meta?.traceId).toBe("t1");
    expect(messages[0]?.meta?.stepsExecuted).toBe(3);
    await waitFor(() => expect(result.current.isStreaming).toBe(false));
  });

  it("degraded=true 时元数据被标记", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"content":"部分答案"}\n\n',
        'data: {"done":true,"status":"degraded","degraded":true,"steps_executed":4,"trace_id":"t2","session_id":"s1"}\n\n',
      ]),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });
    expect(messages[0]?.meta?.degraded).toBe(true);
  });

  it("遇到 awaiting_approval 时挂起并暴露 runId，不算结束", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"content":"先做一半"}\n\n',
        'data: {"awaiting_approval":true,"run_id":"r1","approvals":[{"tool_name":"write"}],"done":true,"status":"awaiting_approval"}\n\n',
      ]),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });

    expect(result.current.pendingApproval?.runId).toBe("r1");
    // 已经渲染的内容不能被丢弃
    expect(messages[0]?.text).toBe("先做一半");
    expect(result.current.isStreaming).toBe(false);
  });

  it("批准后从挂起点续流，且可再次遇到 awaiting_approval（循环）", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"awaiting_approval":true,"run_id":"r1","approvals":[{"tool_name":"write"}],"done":true}\n\n',
      ]),
    );
    vi.spyOn(approvalsApi, "apiResumeRun")
      .mockResolvedValueOnce(
        sseStream([
          'data: {"content":"第一次续跑"}\n\n',
          'data: {"awaiting_approval":true,"run_id":"r1","approvals":[{"tool_name":"delete"}],"done":true}\n\n',
        ]),
      )
      .mockResolvedValueOnce(
        sseStream([
          'data: {"content":"最终答案"}\n\n',
          'data: {"done":true,"status":"success","steps_executed":6,"trace_id":"t3","session_id":"s1","run_id":"r1"}\n\n',
        ]),
      );

    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));

    await act(async () => {
      await result.current.send("q", "s1");
    });
    await act(async () => {
      await result.current.approve({ approved: true, comment: "同意" });
    });
    // 第二轮审批仍应挂起，而不是被当成完成
    expect(result.current.pendingApproval?.runId).toBe("r1");

    await act(async () => {
      await result.current.approve({ approved: true, comment: "" });
    });
    expect(result.current.pendingApproval).toBeNull();
    expect(messages[0]?.text).toBe("第一次续跑最终答案");
    expect(messages[0]?.meta?.stepsExecuted).toBe(6);
  });

  it("拒绝时把 approved:false 与 comment 传给后端", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"awaiting_approval":true,"run_id":"r1","approvals":[],"done":true}\n\n',
      ]),
    );
    const resume = vi
      .spyOn(approvalsApi, "apiResumeRun")
      .mockResolvedValue(
        sseStream(['data: {"done":true,"status":"success"}\n\n']),
      );

    const { onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });
    await act(async () => {
      await result.current.approve({
        approved: false,
        comment: "不要写这个文件",
      });
    });

    expect(resume).toHaveBeenCalledWith(
      "r1",
      { approved: false, comment: "不要写这个文件" },
      expect.anything(),
    );
  });

  it("审批已失效（run_not_found）时给出明确提示且清掉挂起", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"awaiting_approval":true,"run_id":"r1","approvals":[],"done":true}\n\n',
      ]),
    );
    vi.spyOn(approvalsApi, "apiResumeRun").mockResolvedValue(
      sseStream(['data: {"error":"检查点不存在","code":"run_not_found"}\n\n']),
    );

    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });
    await act(async () => {
      await result.current.approve({ approved: true, comment: "" });
    });

    expect(result.current.pendingApproval).toBeNull();
    expect(messages[0]?.error).toMatch(/失效|不存在/);
  });

  it("错误事件不破坏已渲染的正文", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"content":"已经渲染的部分"}\n\n',
        'data: {"error":"模型超时"}\n\n',
      ]),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });
    expect(messages[0]?.text).toBe("已经渲染的部分");
    expect(messages[0]?.error).toBe("模型超时");
  });

  // ---------- 闲聊模式（/chat，非流式 JSON） ----------

  it("闲聊模式走 /chat 一次写入正文，不碰 SSE 接口", async () => {
    const plain = vi.spyOn(chatApi, "sendPlainChat").mockResolvedValue({
      id: "s1",
      model: "m1",
      content: "闲聊回答",
      trace_id: "t9",
    });
    const agent = vi.spyOn(chatApi, "streamAgentChat");

    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("你好", "s1", "chat");
    });

    expect(plain).toHaveBeenCalledWith("你好", "s1", expect.anything());
    expect(agent).not.toHaveBeenCalled();
    expect(messages[0]?.text).toBe("闲聊回答");
    expect(messages[0]?.mode).toBe("chat");
    expect(messages[0]?.meta?.traceId).toBe("t9");
    expect(result.current.isStreaming).toBe(false);
  });

  it("闲聊模式后端不产出步数，脚注不应写'执行 0 步'式的假数据", async () => {
    vi.spyOn(chatApi, "sendPlainChat").mockResolvedValue({
      id: "s1",
      model: "m1",
      content: "回答",
    });
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1", "chat");
    });
    expect(messages[0]?.meta?.stepsExecuted).toBe(0);
  });

  it("闲聊模式出错时记录错误且不丢模式标记", async () => {
    vi.spyOn(chatApi, "sendPlainChat").mockRejectedValue(
      new Error("模型超时"),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1", "chat");
    });
    expect(messages[0]?.error).toBe("模型超时");
    expect(messages[0]?.mode).toBe("chat");
    expect(result.current.isStreaming).toBe(false);
  });

  it("不传模式时默认走工作任务（既有调用方行为不变）", async () => {
    const agent = vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream(['data: {"done":true,"status":"success"}\n\n']),
    );
    const plain = vi.spyOn(chatApi, "sendPlainChat");
    const { onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });
    expect(agent).toHaveBeenCalled();
    expect(plain).not.toHaveBeenCalled();
  });

  it("执行过程（step）逐条累积到当前助手消息上，且不影响正文", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"step":{"node":"execute","tool":"sales_sql_query","title":"查销售额","status":"ok","detail":"12 行"}}\n\n',
        'data: {"step":{"node":"execute","tool":"rag_knowledge_search","title":"查口径","status":"empty_data","detail":"无"}}\n\n',
        'data: {"content":"答案正文"}\n\n',
        'data: {"done":true,"status":"success","steps_executed":2}\n\n',
      ]),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });

    expect(messages[0]?.steps).toHaveLength(2);
    expect(messages[0]?.steps?.[0]?.tool).toBe("sales_sql_query");
    expect(messages[0]?.steps?.[1]?.status).toBe("empty_data");
    // 步骤不能覆盖或污染正文
    expect(messages[0]?.text).toBe("答案正文");
    expect(messages[0]?.meta?.stepsExecuted).toBe(2);
  });
});
