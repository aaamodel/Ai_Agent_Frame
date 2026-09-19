import { useCallback, useEffect, useRef, useState } from "react";

import { useSearchParams } from "react-router-dom";

import { getRunStatus } from "@/api/approvals";
import { Composer } from "@/components/chat/Composer";
import { MessageList } from "@/components/chat/MessageList";

import {
  createSession,
  deriveTitle,
  getSession,
  saveSession,
} from "./sessions";
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
  /**
   * 加载会话时**同步捕获**的待恢复 run_id。
   *
   * ⚠️ 不能等恢复 effect 去读存储：下面的持久化 effect 会用
   *    `pendingRunId: pendingApproval?.runId`（此刻是 undefined）先跑一遍，
   *    把存储里的值抹掉 —— 恢复流程就永远读不到它了。
   */
  const [runIdToRestore, setRunIdToRestore] = useState<string | undefined>(
    () => (urlSession ? getSession(urlSession)?.pendingRunId : undefined),
  );

  // ⚠️ URL 的 session 变化时必须同步状态。
  //   react-router 在**只有查询串变化**时不会重挂载组件，因此从左栏点另一个会话
  //   （/?session=B）时本组件实例不变：不在这里同步的话，用户看到的仍是旧会话内容，
  //   而且新消息会被写进**旧会话**（跨会话串写）。
  useEffect(() => {
    if (!urlSession) {
      const fresh = createSession();
      saveSession(fresh);
      setSessionId(fresh.id);
      setSearchParams({ session: fresh.id }, { replace: true });
      return;
    }
    // 同一批更新，避免出现「sessionId 已换、messages 还是旧的」的中间态被持久化
    const loaded = getSession(urlSession);
    setSessionId(urlSession);
    setMessages(loaded?.messages ?? []);
    setRunIdToRestore(loaded?.pendingRunId);
  }, [urlSession, setSearchParams]);

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

  const { send, approve, abort, restoreApproval, isStreaming, pendingApproval } =
    useChatStream(onMessage);

  // 供"恢复挂起审批"的 effect 读取当前消息，而不必把 messages 放进它的依赖
  // （否则每次消息变化都会重跑回查）。
  const messagesRef = useRef<ChatMessage[]>(messages);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // 组件卸载时中止流：否则连接会一直挂着，且"已中断"标记永远不会出现
  // （设计文档 5.2：中止 fetch，但保留已渲染内容）
  useEffect(() => () => abort(), [abort]);

  // 消息变化即持久化；标题取首条用户消息；同时记录挂起的 run_id 以便刷新后回查
  useEffect(() => {
    if (!sessionId || messages.length === 0) return;
    const firstUser = messages.find((m) => m.role === "user");
    saveSession({
      id: sessionId,
      title: deriveTitle(firstUser?.text ?? ""),
      updatedAt: Date.now(),
      messages,
      pendingRunId: pendingApproval?.runId,
    });
  }, [messages, sessionId, pendingApproval]);

  /**
   * 刷新后恢复挂起的审批（设计文档 5.3）。
   *
   * 没有这段的话：localStorage 里的助手消息带着 `.approval`，刷新后卡片照常渲染，
   * 但 `useChatStream` 是新挂载的、`pendingApproval` 为 null，点"批准"会静默无反应
   * —— 用户看到的是一个死按钮，而那个 run 永远悬着。
   */
  const restoredFor = useRef<string>("");
  useEffect(() => {
    if (!sessionId || !runIdToRestore) return;
    if (restoredFor.current === runIdToRestore) return;
    restoredFor.current = runIdToRestore;
    const runId = runIdToRestore;

    let cancelled = false;
    void (async () => {
      let stillPaused = false;
      try {
        const status = await getRunStatus(runId);
        stillPaused = Boolean(status.exists) && Boolean(status.paused);
      } catch {
        stillPaused = false;
      }
      if (cancelled) return;

      if (stillPaused) {
        // 复用消息里已有的 approvals，保留工具名与参数预览。
        // ⚠️ 在 updater 之外调用 restoreApproval —— 副作用不能写进 setState 的
        //    更新函数里（StrictMode 会执行两次）。
        const existing = messagesRef.current.find((m) => m.approval)?.approval;
        restoreApproval({ runId, approvals: existing?.approvals ?? [] });
        return;
      }
      // 已失效（后端重启或 TTL 过期）：摘掉死卡片并如实说明
      setMessages((prev) =>
        prev.map((m) =>
          m.approval
            ? { ...m, approval: undefined, approvalExpired: true }
            : m,
        ),
      );
    })();

    return () => {
      cancelled = true;
    };
  }, [sessionId, runIdToRestore, restoreApproval]);

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
      <Composer
        // ⚠️ 挂起审批时也要禁用：否则用户能继续提问，而 send() 会清掉挂起状态，
        //    旧卡片随即变成点不动的死按钮，那个 run 就此悬空。
        disabled={isStreaming || Boolean(pendingApproval)}
        onSend={handleSend}
      />
    </div>
  );
}
