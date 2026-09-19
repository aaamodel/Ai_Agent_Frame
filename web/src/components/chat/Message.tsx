import type { ApprovalDecision } from "@/api/types";
import type { ChatMessage } from "@/features/chat/types";

import { ApprovalCard } from "./ApprovalCard";
import { DegradedBanner } from "./DegradedBanner";
import { Markdown } from "./Markdown";
import { MessageMeta } from "./MessageMeta";

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
      <div className="mb-4 flex justify-end">
        <div className="max-w-[62%] rounded-xl bg-neutral-100 px-4 py-2 text-sm">
          {message.text}
        </div>
      </div>
    );
  }

  const showCursor =
    isStreaming && !message.meta && !message.approval && !message.error;

  return (
    <div className="mb-5 flex gap-3">
      <div className="mt-0.5 h-6 w-6 shrink-0 rounded-full bg-blue-600" />
      <div className="min-w-0 flex-1">
        {message.meta?.degraded && <DegradedBanner />}

        {message.text && <Markdown text={message.text} />}
        {showCursor && (
          <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-blue-600 align-text-bottom" />
        )}

        {message.approval && (
          <ApprovalCard
            request={message.approval}
            disabled={isStreaming}
            onDecide={onApprove}
          />
        )}

        {message.approvalExpired && (
          <div className="my-2 rounded border border-neutral-300 bg-neutral-50 px-3 py-2 text-sm text-neutral-600">
            该审批已失效（后端重启或超时），请重新提问。
          </div>
        )}

        {message.interrupted && (
          <div className="mt-2 text-xs text-neutral-400">
            已中断（本轮未走完）
          </div>
        )}

        {message.error && (
          <div className="mt-2 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {message.error}
          </div>
        )}

        {message.meta && <MessageMeta meta={message.meta} />}
      </div>
    </div>
  );
}
