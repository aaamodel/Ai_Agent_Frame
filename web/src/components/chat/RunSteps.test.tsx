import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AgentStep } from "@/features/chat/types";

import { RunSteps } from "./RunSteps";

function step(over: Partial<AgentStep> = {}): AgentStep {
  return {
    node: "execute",
    tool: "sales_sql_query",
    title: "查销售额",
    status: "ok",
    detail: "返回 12 行",
    ...over,
  };
}

describe("RunSteps", () => {
  it("没有步骤但有逐字改写时，仍然渲染改写内容", () => {
    render(<RunSteps steps={[]} running rewriteText="政企优先" />);
    expect(screen.getByText(/政企优先/)).toBeInTheDocument();
  });

  it("既没有步骤也没有改写时不渲染", () => {
    const { container } = render(<RunSteps steps={[]} running={false} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("改写与步骤同时存在时都渲染，互不遮蔽", () => {
    render(
      <RunSteps steps={[step()]} running={false} rewriteText="政企优先" />,
    );
    expect(screen.getByText(/政企优先/)).toBeInTheDocument();
    expect(screen.getByText(/1 步/)).toBeInTheDocument();
    // ⚠️ 必须限定在列表内查工具名：块标题也会显示**最后一步**的工具名，
    //    全局查会命中两个元素（Found multiple elements）。
    expect(
      within(screen.getByRole("list")).getByText("sales_sql_query"),
    ).toBeInTheDocument();
  });

  it("只有改写、没有步骤时，标题不说'0 步'", () => {
    render(<RunSteps steps={[]} running={false} rewriteText="政企优先" />);
    expect(screen.queryByText(/0 步/)).not.toBeInTheDocument();
  });
});
