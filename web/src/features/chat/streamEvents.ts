/** 后端 SSE 事件的前端判别联合。 */

export interface ContentEvent {
  kind: "content";
  text: string;
}

export interface DoneEvent {
  kind: "done";
  status: string;
  degraded: boolean;
  stepsExecuted: number;
  traceId: string;
  sessionId: string;
  runId?: string;
}

/**
 * 步骤事件：Agent 执行过程中的一次工具调用 / 子任务结论。
 *
 * 由后端 `/chat/with_agent` 在图**还在跑**的时候逐条推送（改前这些步骤只存在
 * 于 state.steps 里，整张图跑完才随 done 一起到达，前端全程无反馈）。
 */
export interface StepEvent {
  kind: "step";
  /** 产出该步骤的图节点名（execute / plan / summarize …） */
  node: string;
  tool: string | null;
  title: string;
  status: string;
  detail: string;
}

export interface AwaitingApprovalEvent {
  kind: "awaiting_approval";
  runId: string;
  approvals: unknown[];
}

export interface ErrorEvent {
  kind: "error";
  message: string;
}

export type StreamEvent =
  | ContentEvent
  | DoneEvent
  | AwaitingApprovalEvent
  | ErrorEvent
  | StepEvent;

function str(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}

/**
 * 把一条原始 SSE 事件转成判别联合。
 *
 * ⚠️ 判定顺序很重要：**先判 awaiting_approval，再判 done**。
 * 后端待审批事件同时带 `done: true`，若先判 done 就会把
 * "图已挂起等审批"误当成"本轮正常结束"，对话会永久卡死。
 */
export function toStreamEvent(
  raw: Record<string, unknown>,
): StreamEvent | null {
  if (raw.error) {
    return { kind: "error", message: str(raw.error) };
  }

  // 步骤事件：载荷在 `step` 子对象里，与 done 互斥，顺序上先于 awaiting_approval 判定也无妨
  if (raw.step && typeof raw.step === "object") {
    const s = raw.step as Record<string, unknown>;
    return {
      kind: "step",
      node: str(s.node),
      tool: s.tool ? str(s.tool) : null,
      title: str(s.title),
      status: str(s.status) || "ok",
      detail: str(s.detail),
    };
  }

  if (raw.awaiting_approval) {
    return {
      kind: "awaiting_approval",
      runId: str(raw.run_id),
      approvals: Array.isArray(raw.approvals) ? raw.approvals : [],
    };
  }

  if (raw.done) {
    return {
      kind: "done",
      status: str(raw.status) || "success",
      degraded: raw.degraded === true,
      stepsExecuted: Number(raw.steps_executed ?? 0) || 0,
      traceId: str(raw.trace_id),
      sessionId: str(raw.session_id),
      runId: raw.run_id ? str(raw.run_id) : undefined,
    };
  }

  if (typeof raw.content === "string" && raw.content.length > 0) {
    return { kind: "content", text: raw.content };
  }

  return null;
}
