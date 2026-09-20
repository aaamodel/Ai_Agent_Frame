import { useCallback, useEffect, useRef, useState } from "react";

import { useSearchParams } from "react-router-dom";

import { getRunStatus } from "@/api/approvals";
import { Composer } from "@/components/chat/Composer";
import { MessageList } from "@/components/chat/MessageList";

import { readMode, saveMode } from "./mode";
import {
  createSession,
  deriveTitle,
  getSession,
  saveSession,
} from "./sessions";
import type { ChatMessage, ChatMode } from "./types";
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

  /**
   * 对话模式（闲聊 / 工作任务）。
   *
   * 与会话历史分开存：模式是**偏好**，不该绑死在某一个会话上——
   * 切到旧会话时沿用当前偏好，和豆包的交互一致。
   *
   * ⚠️ 必须声明在 handleSend **之前**：handleSend 的依赖数组引用了 mode，
   *   放到后面会触发 TDZ（"used before its declaration"），整个组件直接崩。
   */
  const [mode, setMode] = useState<ChatMode>(() => readMode());

  function changeMode(next: ChatMode) {
    setMode(next);
    saveMode(next);
  }

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
      // 发送时读取当时的模式：切模式只影响之后的轮次，不改写历史
      void send(text, sessionId, mode);
    },
    [send, sessionId, mode],
  );

  /** 纯 UI 状态：设置面板开关。不参与会话持久化。 */
  const [showSettings, setShowSettings] = useState(false);

  const title = deriveTitle(messages.find((m) => m.role === "user")?.text ?? "");

  // Agent 状态：等待审批 > 运行中 > 空闲（审批态优先——它才是需要用户动作的那一档）
  const status = pendingApproval
    ? { label: "等待审批", dot: "bg-warn-text agent-pulse", text: "text-warn-text" }
    : isStreaming
      ? { label: "运行中", dot: "bg-accent agent-pulse", text: "text-accent-text" }
      : { label: "空闲", dot: "bg-fg-subtle", text: "text-fg-subtle" };

  function clearMessages() {
    setMessages([]);
    setShowSettings(false);
    if (!sessionId) return;
    // 同步落盘为空会话：否则刷新后旧消息又会被读回来，"清空"等于没清
    saveSession({
      id: sessionId,
      title: "新会话",
      updatedAt: Date.now(),
      messages: [],
    });
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* 顶部标题栏：会话名 + Agent 状态 + 操作 */}
      <header className="relative flex h-14 shrink-0 items-center gap-2 border-b border-line bg-surface-1 px-4">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[16px] leading-tight font-semibold text-fg">
            {title}
          </div>
          <div className="mt-0.5 flex items-center gap-1.5 text-[11px]">
            <span className={`h-1.5 w-1.5 rounded-full ${status.dot}`} />
            <span className={status.text}>{status.label}</span>
          </div>
        </div>

        <button
          onClick={clearMessages}
          title="清空当前会话"
          aria-label="清空当前会话"
          className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-fg-subtle transition-colors hover:bg-surface-2 hover:text-danger-text"
        >
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.8}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v5M14 11v5" />
          </svg>
        </button>

        <button
          onClick={() => setShowSettings((v) => !v)}
          title="会话信息"
          aria-label="会话信息"
          className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-fg-subtle transition-colors hover:bg-surface-2 hover:text-fg"
        >
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.8}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.24.6.86 1.01 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>

        {showSettings && (
          <div className="absolute right-3 top-full z-20 w-64 rounded-lg border border-line bg-surface-2 p-3 shadow-xl">
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-fg-subtle">
              会话 ID
            </div>
            <div className="break-all rounded-md bg-surface-3 px-2 py-1.5 font-mono text-[11px] text-fg-muted">
              {sessionId || "—"}
            </div>
            <div className="mt-2 text-[11px] text-fg-subtle">
              会话历史保存在本机浏览器；后端不提供会话列表接口。
            </div>
          </div>
        )}
      </header>

      <MessageList
        messages={messages}
        isStreaming={isStreaming}
        onApprove={approve}
        onPickExample={handleSend}
      />
      <Composer
        // ⚠️ 挂起审批时也要禁用：否则用户能继续提问，而 send() 会清掉挂起状态，
        //    旧卡片随即变成点不动的死按钮，那个 run 就此悬空。
        disabled={isStreaming || Boolean(pendingApproval)}
        isStreaming={isStreaming}
        onStop={abort}
        onSend={handleSend}
        mode={mode}
        onModeChange={changeMode}
      />
    </div>
  );
}
