export type ChatRole = "user" | "assistant";

export interface MessageMeta {
  status: string;
  degraded: boolean;
  stepsExecuted: number;
  traceId: string;
  sessionId: string;
}

export interface ApprovalRequest {
  runId: string;
  approvals: unknown[];
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  meta?: MessageMeta;
  /** 流式被中断（用户切页/卸载），这一轮没有正常走完 */
  interrupted?: boolean;
  error?: string;
  /** 该助手消息上挂着一个待审批项 */
  approval?: ApprovalRequest;
  /** 该消息对应的审批已失效 */
  approvalExpired?: boolean;
}
