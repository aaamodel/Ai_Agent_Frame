import { useEffect, useRef } from "react";

import type { ApprovalDecision } from "@/api/types";
import type { ChatMessage } from "@/features/chat/types";

import { Message } from "./Message";

/** 空状态里的快捷示例问题：直接可点，省掉首次使用的空白焦虑。 */
const EXAMPLES = [
  "我们优先做哪些行业？哪些算次优先？",
  "本月销售额排名前三的客户是哪些？",
  "华东区上季度的回款情况怎么样？",
];

function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-6 py-10">
      <div className="mb-4 grid h-14 w-14 place-items-center rounded-2xl border border-line bg-surface-2">
        <svg
          viewBox="0 0 24 24"
          className="h-6 w-6 text-accent-text"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.6}
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M12 3l2.2 5.6L20 11l-5.8 2.4L12 19l-2.2-5.6L4 11l5.8-2.4z" />
        </svg>
      </div>
      <div className="mb-1 text-[16px] font-semibold text-fg">
        开始你的第一轮对话
      </div>
      <div className="mb-6 text-xs text-fg-subtle">
        从下面的问题开始，或直接在下方输入
      </div>
      <div className="flex w-full max-w-md flex-col gap-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            className="rounded-lg border border-line bg-surface-2 px-3 py-2.5 text-left text-[13px] text-fg-muted transition-colors hover:border-accent-ring hover:bg-surface-3 hover:text-fg"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

export function MessageList({
  messages,
  isStreaming,
  onApprove,
  onPickExample,
}: {
  messages: ChatMessage[];
  isStreaming: boolean;
  onApprove: (d: ApprovalDecision) => void;
  onPickExample?: (text: string) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // 新内容到达时滚到底（流式回答时持续生效）
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages, isStreaming]);

  if (messages.length === 0) {
    return (
      <div
        data-testid="message-list"
        className="flex min-h-0 flex-1 flex-col overflow-y-auto"
      >
        <EmptyState
          onPick={(q) => onPickExample?.(q)}
        />
      </div>
    );
  }

  return (
    <div data-testid="message-list" className="min-h-0 flex-1 overflow-y-auto">
      {/* 内容限宽居中：超宽屏上正文不会拉成一行到底，可读性更好 */}
      <div className="mx-auto w-full max-w-3xl px-6 py-6">
        {messages.map((m) => (
          <Message
            key={m.id}
            message={m}
            isStreaming={isStreaming}
            onApprove={onApprove}
          />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
