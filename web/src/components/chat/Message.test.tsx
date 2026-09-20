import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ChatMessage } from "@/features/chat/types";

import { Message } from "./Message";

function msg(over: Partial<ChatMessage> = {}): ChatMessage {
  return { id: "m1", role: "assistant", text: "回答正文", ...over };
}

describe("Message", () => {
  it("降级时显示显式警示条", () => {
    render(
      <Message
        message={msg({
          meta: {
            status: "degraded",
            degraded: true,
            stepsExecuted: 4,
            traceId: "t",
            sessionId: "s",
          },
        })}
        isStreaming={false}
        onApprove={vi.fn()}
      />,
    );
    expect(screen.getByText(/未取全信息/)).toBeInTheDocument();
  });

  it("成功时只留脚注，不出现警示条", () => {
    render(
      <Message
        message={msg({
          meta: {
            status: "success",
            degraded: false,
            stepsExecuted: 3,
            traceId: "abc123",
            sessionId: "s",
          },
        })}
        isStreaming={false}
        onApprove={vi.fn()}
      />,
    );
    expect(screen.queryByText(/未取全信息/)).not.toBeInTheDocument();
    expect(screen.getByText(/执行 3 步/)).toBeInTheDocument();
    expect(screen.getByText(/abc123/)).toBeInTheDocument();
  });

  it("中断的那一轮被标记，不伪装成正常完成", () => {
    render(
      <Message
        message={msg({ interrupted: true })}
        isStreaming={false}
        onApprove={vi.fn()}
      />,
    );
    expect(screen.getByText(/已中断/)).toBeInTheDocument();
  });

  it("错误消息与已有正文同时存在", () => {
    render(
      <Message
        message={msg({ error: "模型超时" })}
        isStreaming={false}
        onApprove={vi.fn()}
      />,
    );
    expect(screen.getByText("回答正文")).toBeInTheDocument();
    expect(screen.getByText(/模型超时/)).toBeInTheDocument();
  });

  it("审批失效时给出可理解的说明", () => {
    render(
      <Message
        message={msg({ approvalExpired: true })}
        isStreaming={false}
        onApprove={vi.fn()}
      />,
    );
    expect(screen.getByText(/审批已失效/)).toBeInTheDocument();
  });

  // ---------- 换候选废弃段落：保留并标注，不静默丢弃 ----------

  it("被废弃的半截答案仍然可见，且带明确标注", () => {
    render(
      <Message
        message={msg({
          text: "第二候选的答案",
          abandoned: ["第一候选的开头"],
        })}
        isStreaming={false}
        onApprove={vi.fn()}
      />,
    );
    // 废弃内容没有被丢掉
    expect(screen.getByText("第一候选的开头")).toBeInTheDocument();
    // 且明确标注了作废原因，用户不会以为答案凭空换了
    expect(screen.getByText(/上段因模型切换已废弃/)).toBeInTheDocument();
    // 当前答案照常显示
    expect(screen.getByText("第二候选的答案")).toBeInTheDocument();
  });

  it("没有废弃段落时不渲染标注", () => {
    render(
      <Message
        message={msg()}
        isStreaming={false}
        onApprove={vi.fn()}
      />,
    );
    expect(screen.queryByText(/上段因模型切换已废弃/)).not.toBeInTheDocument();
  });
});
