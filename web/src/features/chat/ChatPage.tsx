import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { Composer } from "@/components/chat/Composer";
import { MessageList } from "@/components/chat/MessageList";

import { createSession, deriveTitle, getSession, saveSession } from "./sessions";
import type { ChatMessage } from "./types";
import { useChatStream } from "./useChatStream";

/**
 * 对话页。
 *
 * 会话由 URL 上的 `?session=<id>` 标识（见设计文档第 9 节）：
 *   - 带参数 → 从本地存储恢复该会话的历史
 *   - 不带参数 → 立刻建一个新会话并写回 URL（replace，不污染浏览器历史）
 *
 * 消息内容由本页保管、经 onMessage 交给状态机更新——这样用户切页时
 * 已渲染内容不会随组件卸载消失。
 */
export function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlSession = searchParams.get("session") ?? "";

  const [sessionId, setSessionId] = useState(urlSession);
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    urlSession ? (getSession(urlSession)?.messages ?? []) : [],
  );

  // URL 没带 session：建一个并写回去
  useEffect(() => {
    if (sessionId) return;
    const fresh = createSession();
    saveSession(fresh);
    setSessionId(fresh.id);
    setSearchParams({ session: fresh.id }, { replace: true });
  }, [sessionId, setSearchParams]);

  /** 把更新作用到"当前这条"助手消息上（页面负责压入占位消息）。 */
  const onMessage = useCallback(
    (updater: (m: ChatMessage) => ChatMessage) => {
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const next = [...prev];
        const last = next.length - 1;
        next[last] = updater(next[last]);
        return next;
      });
    },
    [],
  );

  const { send, approve, isStreaming } = useChatStream(onMessage);

  // 消息变化即持久化；标题取首条用户消息
  useEffect(() => {
    if (!sessionId || messages.length === 0) return;
    const firstUser = messages.find((m) => m.role === "user");
    saveSession({
      id: sessionId,
      title: deriveTitle(firstUser?.text ?? ""),
      updatedAt: Date.now(),
      messages,
    });
  }, [messages, sessionId]);

  const handleSend = useCallback(
    (text: string) => {
      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        text,
      };
      // 先压入助手占位消息：状态机只更新"最后一条"，它不会自己创建消息
      const placeholder: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: "",
      };
      setMessages((prev) => [...prev, userMessage, placeholder]);
      void send(text, sessionId);
    },
    [send, sessionId],
  );

  return (
    <div className="flex h-full flex-col">
      <MessageList
        messages={messages}
        isStreaming={isStreaming}
        onApprove={approve}
      />
      <Composer disabled={isStreaming} onSend={handleSend} />
    </div>
  );
}
