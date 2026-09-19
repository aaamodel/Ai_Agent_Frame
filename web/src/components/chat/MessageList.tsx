import { useEffect, useRef } from "react";

import type { ApprovalDecision } from "@/api/types";
import type { ChatMessage } from "@/features/chat/types";

import { Message } from "./Message";

export function MessageList({
  messages,
  isStreaming,
  onApprove,
}: {
  messages: ChatMessage[];
  isStreaming: boolean;
  onApprove: (d: ApprovalDecision) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // 新内容到达时滚到底（流式回答时持续生效）
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages, isStreaming]);

  return (
    <div className="flex-1 overflow-y-auto px-6 py-5">
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
  );
}
