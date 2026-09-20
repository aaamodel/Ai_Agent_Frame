import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { Sidebar } from "./Sidebar";

describe("Sidebar", () => {
  it("三个区导航都在", () => {
    render(
      <MemoryRouter>
        <Sidebar pendingCount={0} pendingItems={[]} />
      </MemoryRouter>,
    );
    expect(screen.getByText(/对话/)).toBeInTheDocument();
    expect(screen.getByText(/文档/)).toBeInTheDocument();
    expect(screen.getByText(/知识库/)).toBeInTheDocument();
  });

  it("有待审批时不显示角标；有条数时显示数字", () => {
    const { rerender } = render(
      <MemoryRouter>
        <Sidebar pendingCount={0} pendingItems={[]} />
      </MemoryRouter>,
    );
    // 用正则而非精确串：角标那行的直接文本是"⚠ 待审批"，精确匹配会漏
    expect(screen.queryByText(/待审批/)).not.toBeInTheDocument();

    rerender(
      <MemoryRouter>
        <Sidebar
          pendingCount={2}
          pendingItems={[
            { run_id: "r1", session_id: "s1", query: "问一句" } as never,
          ]}
        />
      </MemoryRouter>,
    );
    expect(screen.getByText(/待审批/)).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("内存后端时如实提示重启会失效", () => {
    render(
      <MemoryRouter>
        <Sidebar pendingCount={1} pendingItems={[]} backend="memory" />
      </MemoryRouter>,
    );
    expect(screen.getByText(/后端重启/)).toBeInTheDocument();
  });
});
