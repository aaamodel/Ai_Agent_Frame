import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import * as chatApi from "@/api/chat";

import { ChatPage } from "./ChatPage";
import { createSession, listSessions, saveSession } from "./sessions";

function renderChat(initialEntry = "/") {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
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
      new ReadableStream<Uint8Array>({
        start(c) {
          c.close();
        },
      }),
    );
    renderChat("/");
    await userEvent.type(screen.getByPlaceholderText(/问点什么/), "行业优先级");
    await userEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(listSessions()[0]?.title).toBe("行业优先级"));
  });
});
