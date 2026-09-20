import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ApprovalCard } from "./ApprovalCard";

describe("ApprovalCard", () => {
  it("批准时回传 approved:true 与备注", async () => {
    const onDecide = vi.fn();
    render(
      <ApprovalCard
        request={{
          runId: "r1",
          approvals: [{ tool_name: "local_excel_write_tool" }],
        }}
        disabled={false}
        onDecide={onDecide}
      />,
    );
    await userEvent.type(screen.getByPlaceholderText(/备注/), "同意");
    await userEvent.click(screen.getByRole("button", { name: "批准" }));
    expect(onDecide).toHaveBeenCalledWith({ approved: true, comment: "同意" });
  });

  it("拒绝时回传 approved:false", async () => {
    const onDecide = vi.fn();
    render(
      <ApprovalCard
        request={{ runId: "r1", approvals: [] }}
        disabled={false}
        onDecide={onDecide}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: "拒绝" }));
    expect(onDecide).toHaveBeenCalledWith({ approved: false, comment: "" });
  });

  it("处理中时按钮禁用，防止重复提交", () => {
    render(
      <ApprovalCard
        request={{ runId: "r1", approvals: [] }}
        disabled
        onDecide={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: "批准" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "拒绝" })).toBeDisabled();
  });
});
