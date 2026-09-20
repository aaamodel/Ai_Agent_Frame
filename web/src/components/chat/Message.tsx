import type { ApprovalDecision } from "@/api/types";
import type { ChatMessage } from "@/features/chat/types";

import { ApprovalCard } from "./ApprovalCard";
import { DegradedBanner } from "./DegradedBanner";
import { Markdown } from "./Markdown";
import { MessageMeta } from "./MessageMeta";
import { RunSteps } from "./RunSteps";

export function Message({
  message,
  isStreaming,
  onApprove,
}: {
  message: ChatMessage;
  isStreaming: boolean;
  onApprove: (d: ApprovalDecision) => void;
}) {
  if (message.role === "user") {
    // 用户消息：右对齐浅色块（全宽文档式，非气泡）
    return (
      <div className="mb-6 flex justify-end">
        <div className="max-w-[62%] rounded-lg bg-surface-3 px-4 py-2.5 text-sm whitespace-pre-wrap break-words">
          {message.text}
        </div>
      </div>
    );
  }

  const showCursor =
    isStreaming && !message.meta && !message.approval && !message.error;

  return (
    <div className="mb-6 flex gap-3">
      {/* 助手头像：与主 Logo 同构，作为"这是 Agent"的视觉锚点 */}
      <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-accent-soft text-[11px] font-bold text-accent-text">
        A
      </div>
      {/* 助手正文占满剩余宽度：长回复 / 代码块 / 表格不塞进窄气泡 */}
      <div className="min-w-0 flex-1">
        {/* 执行过程：答案出来之前唯一的进展反馈 */}
        <RunSteps steps={message.steps ?? []} running={showCursor} />

        {/* 闲聊模式产生的回答：标注来源，和工作任务的回答区分开 */}
        {message.mode === "chat" && (
          <div className="mb-1.5 flex">
            <span className="rounded-md border border-line bg-surface-2 px-1.5 py-0.5 text-[10px] text-fg-subtle">
              闲聊模式
            </span>
          </div>
        )}

        {message.meta?.degraded && <DegradedBanner />}

        {message.text && <Markdown text={message.text} />}
        {showCursor && <span className="agent-caret" aria-hidden="true" />}

        {message.approval && (
          <ApprovalCard
            request={message.approval}
            disabled={isStreaming}
            onDecide={onApprove}
          />
        )}

        {message.approvalExpired && (
          <div className="my-2 rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-fg-muted">
            该审批已失效（后端重启或超时），请重新提问。
          </div>
        )}

        {message.interrupted && (
          <div className="mt-2 text-xs text-fg-subtle">
            已中断（本轮未走完）
          </div>
        )}

        {message.error && (
          <div className="mt-2 rounded-lg border border-danger-border bg-danger-bg px-3 py-2 text-sm text-danger-text">
            {message.error}
          </div>
        )}

        {message.meta && <MessageMeta meta={message.meta} />}
      </div>
    </div>
  );
}
