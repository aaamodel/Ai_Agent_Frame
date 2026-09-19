import { useCallback, useRef, useState } from "react";

import { apiResumeRun } from "@/api/approvals";
import { streamAgentChat } from "@/api/chat";
import type { ApprovalDecision } from "@/api/types";
import { parseSSEStream } from "@/lib/sse";

import { toStreamEvent } from "./streamEvents";
import type { ApprovalRequest, ChatMessage } from "./types";

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

  const send = useCallback(
    async (query: string, sessionId: string) => {
      setIsStreaming(true);
      setPendingApproval(null);
      const controller = new AbortController();
      abortRef.current = controller;
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

  return { send, approve, abort, isStreaming, pendingApproval };
}
