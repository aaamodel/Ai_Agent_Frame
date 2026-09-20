import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import type { Session } from "@/features/chat/sessions";

import { SessionListPane } from "./ContextPanes";

function session(over: Partial<Session> = {}): Session {
  return {
    id: "s1",
    title: "行业优先级",
    updatedAt: Date.now(),
    messages: [],
    ...over,
  };
}

function renderPane(sessions: Session[], onDelete?: (s: Session) => void) {
  return render(
    <MemoryRouter>
      <SessionListPane sessions={sessions} onDelete={onDelete} />
    </MemoryRouter>,
  );
}

describe("SessionListPane", () => {
  it("点删除按钮把该会话交回调用方", async () => {
    const onDelete = vi.fn();
    const s = session();
    renderPane([s], onDelete);

    await userEvent.click(
      screen.getByRole("button", { name: `删除会话 ${s.title}` }),
    );

    expect(onDelete).toHaveBeenCalledWith(s);
  });

  /**
   * ⚠️ 回归：删除按钮**不能**嵌在 `<a>` 里。
   *
   * 嵌进去的话，点击按钮的事件会冒泡到链接上——删完之后还会顺手跳转到那个
   * 刚被删掉的会话，界面停在一条不存在的会话上。
   */
  it("删除按钮不在链接内部", () => {
    renderPane([session()], vi.fn());

    const button = screen.getByRole("button", { name: /删除会话/ });
    expect(button.closest("a")).toBeNull();
  });

  it("不传 onDelete 时不渲染删除按钮", () => {
    renderPane([session()]);

    expect(screen.queryByRole("button", { name: /删除会话/ })).toBeNull();
  });

  it("没有会话时给出引导文案", () => {
    renderPane([], vi.fn());

    expect(screen.getByText(/还没有会话/)).toBeInTheDocument();
  });
});
