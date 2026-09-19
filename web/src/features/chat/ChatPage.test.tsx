import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import * as approvalsApi from "@/api/approvals";
import * as chatApi from "@/api/chat";

import { ChatPage } from "./ChatPage";
import { createSession, listSessions, saveSession } from "./sessions";

function sseStream(chunks: string[]) {
  const enc = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(c) {
      for (const s of chunks) c.enqueue(enc.encode(s));
      c.close();
    },
  });
}

function renderChat(initialEntry = "/") {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <ChatPage />
    </MemoryRouter>,
  );
}

/** 带一个跳转链接的宿主，用来模拟"在左栏点另一个会话"。 */
function renderWithSwitcher(initialEntry: string, target: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Link to={target}>去另一个会话</Link>
      <ChatPage />
    </MemoryRouter>,
  );
}

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe("ChatPage", () => {
  it("首次进入会创建一个会话并持久化", async () => {
    renderChat("/");
    expect(screen.getByPlaceholderText(/问点什么/)).toBeInTheDocument();
    await waitFor(() => expect(listSessions().length).toBe(1));
  });

  it("发送后出现用户消息，并调用流式接口", async () => {
    const spy = vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      new ReadableStream<Uint8Array>({
        start(c) {
          c.close();
        },
      }),
    );
    renderChat("/");

    await userEvent.type(
      screen.getByPlaceholderText(/问点什么/),
      "优先做哪些行业",
    );
    await userEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(screen.getByText("优先做哪些行业")).toBeInTheDocument();
  });

  it("带 session 参数时从本地恢复历史消息", async () => {
    const s = {
      ...createSession(),
      messages: [{ id: "m1", role: "user" as const, text: "历史提问" }],
    };
    saveSession(s);

    renderChat(`/?session=${s.id}`);
    await waitFor(() =>
      expect(screen.getByText("历史提问")).toBeInTheDocument(),
    );
  });

  it("发送后会话标题取自首条用户消息", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([]), // 立即结束的空流
    );
    renderChat("/");
    await userEvent.type(screen.getByPlaceholderText(/问点什么/), "行业优先级");
    await userEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(listSessions()[0]?.title).toBe("行业优先级"));
  });

  /**
   * Critical：react-router 只改查询串时**不会重挂载**组件，
   * 因此从左栏点另一个会话时必须主动同步状态——否则 B 的内容永远加载不出来，
   * 而且新消息会被写进 A（跨会话串写）。
   */
  it("切换 session 参数时加载对应会话，且不把消息写进旧会话", async () => {
    const a = {
      ...createSession(),
      messages: [{ id: "a1", role: "user" as const, text: "会话A的问题" }],
    };
    const b = {
      ...createSession(),
      messages: [{ id: "b1", role: "user" as const, text: "会话B的问题" }],
    };
    saveSession(a);
    saveSession(b);

    renderWithSwitcher(`/?session=${a.id}`, `/?session=${b.id}`);
    await waitFor(() =>
      expect(screen.getByText("会话A的问题")).toBeInTheDocument(),
    );

    await userEvent.click(screen.getByText("去另一个会话"));

    await waitFor(() =>
      expect(screen.getByText("会话B的问题")).toBeInTheDocument(),
    );
    expect(screen.queryByText("会话A的问题")).not.toBeInTheDocument();
  });

  /**
   * Critical：刷新后 localStorage 里的消息带着 .approval，卡片照常渲染，
   * 但状态机是新挂载的、pendingApproval 为 null → 点批准会静默无反应。
   * 必须按 pendingRunId 回查，仍挂起则把状态接管回来。
   */
  it("刷新后按 pendingRunId 回查，仍挂起则重建审批且批准真的生效", async () => {
    const s = {
      ...createSession(),
      pendingRunId: "r1",
      messages: [
        { id: "m1", role: "user" as const, text: "帮我写文件" },
        {
          id: "m2",
          role: "assistant" as const,
          text: "",
          approval: {
            runId: "r1",
            approvals: [{ tool_name: "local_excel_write_tool" }],
          },
        },
      ],
    };
    saveSession(s);
    vi.spyOn(approvalsApi, "getRunStatus").mockResolvedValue({
      exists: true,
      paused: true,
    });
    const resume = vi
      .spyOn(approvalsApi, "apiResumeRun")
      .mockResolvedValue(sseStream(['data: {"done":true,"status":"success"}\n\n']));

    renderChat(`/?session=${s.id}`);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "批准" })).toBeInTheDocument(),
    );

    await userEvent.click(screen.getByRole("button", { name: "批准" }));
    await waitFor(() => expect(resume).toHaveBeenCalledWith("r1", expect.anything(), expect.anything()));
  });

  it("刷新后若审批已失效，摘掉死卡片并如实说明", async () => {
    const s = {
      ...createSession(),
      pendingRunId: "r1",
      messages: [
        {
          id: "m1",
          role: "assistant" as const,
          text: "部分回答",
          approval: { runId: "r1", approvals: [] },
        },
      ],
    };
    saveSession(s);
    vi.spyOn(approvalsApi, "getRunStatus").mockResolvedValue({ exists: false });

    renderChat(`/?session=${s.id}`);
    await waitFor(() =>
      expect(screen.getByText(/审批已失效/)).toBeInTheDocument(),
    );
    expect(screen.queryByRole("button", { name: "批准" })).not.toBeInTheDocument();
  });

  /** Important：挂起审批时还能发新问题，会把挂起项变成孤儿死按钮。 */
  it("挂起审批期间发送按钮被禁用", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"awaiting_approval":true,"run_id":"r1","approvals":[{"tool_name":"write"}],"done":true}\n\n',
      ]),
    );
    renderChat("/");
    await userEvent.type(screen.getByPlaceholderText(/问点什么/), "q");
    await userEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "批准" })).toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: "发送" })).toBeDisabled();
  });

  /** Important：卸载时必须中止流，否则连接挂着且"已中断"永不出现。 */
  it("卸载时中止进行中的流", async () => {
    let captured: AbortSignal | undefined;
    // 用持有对象而不是裸 let：TS 的控制流分析看不到闭包里的赋值，
    // 会把裸变量收窄成 never，导致 ctrl?.close() 报 TS2339。
    const holder: { ctrl?: ReadableStreamDefaultController<Uint8Array> } = {};
    vi.spyOn(chatApi, "streamAgentChat").mockImplementation(
      async (_q, _s, signal) => {
        captured = signal;
        return new ReadableStream<Uint8Array>({
          start(c) {
            holder.ctrl = c;
          },
        });
      },
    );

    const { unmount } = renderChat("/");
    await userEvent.type(screen.getByPlaceholderText(/问点什么/), "q");
    await userEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(captured).toBeDefined());
    expect(captured?.aborted).toBe(false);

    unmount();
    expect(captured?.aborted).toBe(true);
    holder.ctrl?.close();
  });
});
