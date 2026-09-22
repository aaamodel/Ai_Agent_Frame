import { useCallback, useRef, useState } from "react";

import { apiResumeRun } from "@/api/approvals";
import { sendPlainChat, streamAgentChat } from "@/api/chat";
import type { ApprovalDecision } from "@/api/types";
import { parseSSEStream } from "@/lib/sse";

import { toStreamEvent } from "./streamEvents";
import type { ApprovalRequest, ChatMessage, ChatMode } from "./types";

export type MessageUpdater = (updater: (m: ChatMessage) => ChatMessage) => void;

/**
 * 对话流状态机。
 *
 * 状态：idle → streaming → (awaiting_approval ⇄ streaming) → done | error
 *
 * ⚠️ 审批是**循环**：续跑中再次命中危险工具会再次推 awaiting_approval
 * （同一个 run_id），因此不能在批准后就把状态清成"结束"。
 *
 * ⚠️ 内容不在这里保管，而是通过 `onMessage` 交给页面的消息列表——
 * 这样用户切页/组件卸载时已渲染内容不会随 hook 一起消失。
 */
export function useChatStream(onMessage: MessageUpdater) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [pendingApproval, setPendingApproval] =
    useState<ApprovalRequest | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  /** 消费一条 SSE 流，把事件作用到当前助手消息上。 */
  const consume = useCallback(
    async (stream: ReadableStream<Uint8Array>) => {
      let sawApproval: ApprovalRequest | null = null;
      let sawError = "";

      for await (const raw of parseSSEStream(stream)) {
        const ev = toStreamEvent(raw);
        if (!ev) continue;

        if (ev.kind === "content") {
          onMessage((m) => ({ ...m, text: m.text + ev.text }));
        } else if (ev.kind === "step") {
          // 执行过程：累积到当前助手消息上，用户能在答案出来之前就看到进展
          onMessage((m) => ({
            ...m,
            steps: [
              ...(m.steps ?? []),
              {
                node: ev.node,
                tool: ev.tool,
                title: ev.title,
                status: ev.status,
                detail: ev.detail,
              },
            ],
          }));
        } else if (ev.kind === "delta") {
          // 逐字增量：改写进 rewriteText，答案进正文 —— 两条缓冲互相独立
          onMessage((m) => {
            if (ev.phase === "rewrite") {
              if (ev.attemptReset) {
                // 改写 LLM 重试/换候选：旧尝试已逐字显示的半截改写作废，
                // 直接清空（rewriteText 只是过程性回显，无需像正文那样保留划线）
                return m.rewriteText ? { ...m, rewriteText: "" } : m;
              }
              return { ...m, rewriteText: (m.rewriteText ?? "") + ev.text };
            }
            if (ev.attemptReset) {
              // 换候选/重试：旧的半截答案作废。按"保留并标注"策略移到
              // abandoned，正文从空重新开始——不静默丢弃。
              return m.text
                ? { ...m, abandoned: [...(m.abandoned ?? []), m.text], text: "" }
                : m;
            }
            return { ...m, text: m.text + ev.text };
          });
        } else if (ev.kind === "awaiting_approval") {
          // 记录挂起，等用户决策；**不能**在这里当成结束
          sawApproval = { runId: ev.runId, approvals: ev.approvals };
        } else if (ev.kind === "done") {
          onMessage((m) => ({
            ...m,
            meta: {
              status: ev.status,
              degraded: ev.degraded,
              stepsExecuted: ev.stepsExecuted,
              traceId: ev.traceId,
              sessionId: ev.sessionId,
            },
          }));
        } else if (ev.kind === "error") {
          sawError = ev.message;
        }
      }

      return { sawApproval, sawError };
    },
    [onMessage],
  );

  /** 依据一轮消费结果落定状态。 */
  const settle = useCallback(
    (outcome: { sawApproval: ApprovalRequest | null; sawError: string }) => {
      if (outcome.sawError) {
        const msg = outcome.sawError;
        onMessage((m) => ({
          ...m,
          error: /不存在|检查点/.test(msg) ? "该审批已失效（后端重启或超时）" : msg,
        }));
        setPendingApproval(null);
        setIsStreaming(false);
        return;
      }
      if (outcome.sawApproval) {
        const req = outcome.sawApproval;
        onMessage((m) => ({ ...m, approval: req }));
        setPendingApproval(req);
        setIsStreaming(false);
        return;
      }
      onMessage((m) => ({ ...m, approval: undefined }));
      setPendingApproval(null);
      setIsStreaming(false);
    },
    [onMessage],
  );

  /**
   * 发送一轮对话。
   *
   * `mode = "agent"`（默认）走 `/chat/with_agent` 的 SSE 流；
   * `mode = "chat"` 闲聊走 `/chat` 的一次性 JSON——没有审批、没有降级、
   * 也没有步数（后端这个端点不产出这些字段）。
   *
   * ⚠️ 默认值 `agent` 是刻意的：既有的调用方与测试都只传两个参数，
   *   不应因为新增模式而改变它们的行为。
   */
  const send = useCallback(
    async (query: string, sessionId: string, mode: ChatMode = "agent") => {
      setIsStreaming(true);
      setPendingApproval(null);
      const controller = new AbortController();
      abortRef.current = controller;

      // 在占位消息上先记下模式，界面上才知道这条回答来自哪条链路
      onMessage((m) => ({ ...m, mode }));

      // ---------- 闲聊：一次性 JSON，没有流也没有审批 ----------
      if (mode === "chat") {
        try {
          const resp = await sendPlainChat(query, sessionId, controller.signal);
          onMessage((m) => ({
            ...m,
            text: resp.content,
            meta: {
              status: "success",
              degraded: false,
              stepsExecuted: 0,
              traceId: resp.trace_id ?? "",
              // 后端把 session_id 复用在 id 字段里返回
              sessionId: resp.id || sessionId,
            },
          }));
          setIsStreaming(false);
        } catch (e) {
          if (controller.signal.aborted) {
            onMessage((m) => ({ ...m, interrupted: true }));
            setIsStreaming(false);
            return;
          }
          const message = e instanceof Error ? e.message : String(e);
          onMessage((m) => ({ ...m, error: message }));
          setIsStreaming(false);
        }
        return;
      }

      // ---------- 工作任务：SSE 流 ----------
      try {
        const stream = await streamAgentChat(
          query,
          sessionId,
          controller.signal,
        );
        settle(await consume(stream));
      } catch (e) {
        if (controller.signal.aborted) {
          // 用户主动切走：内容留着，但这一轮标记为已中断，不假装完成
          onMessage((m) => ({ ...m, interrupted: true }));
          setIsStreaming(false);
          return;
        }
        const message = e instanceof Error ? e.message : String(e);
        onMessage((m) => ({ ...m, error: message }));
        setIsStreaming(false);
      }
    },
    [consume, onMessage, settle],
  );

  const approve = useCallback(
    async (decision: ApprovalDecision) => {
      const current = pendingApproval;
      if (!current) return;
      setIsStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const stream = await apiResumeRun(
          current.runId,
          decision,
          controller.signal,
        );
        settle(await consume(stream));
      } catch (e) {
        if (controller.signal.aborted) {
          onMessage((m) => ({ ...m, interrupted: true }));
          setIsStreaming(false);
          return;
        }
        const message = e instanceof Error ? e.message : String(e);
        onMessage((m) => ({ ...m, error: message }));
        setPendingApproval(null);
        setIsStreaming(false);
      }
    },
    [consume, onMessage, pendingApproval, settle],
  );

  const abort = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  /**
   * 重新接管一个**已存在的**挂起审批（刷新后按 run_id 回查恢复用）。
   *
   * 没有它的话，页面刷新后卡片虽然从 localStorage 渲染出来了，
   * 但 `pendingApproval` 是 null，`approve()` 会在第一行静默返回 ——
   * 用户看到一个点不动的"批准"按钮，且那个 run 永远悬着。
   */
  const restoreApproval = useCallback((req: ApprovalRequest) => {
    setPendingApproval(req);
  }, []);

  return { send, approve, abort, restoreApproval, isStreaming, pendingApproval };
}
