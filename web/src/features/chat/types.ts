export type ChatRole = "user" | "assistant";

/**
 * 对话模式。
 *
 * - `chat`  闲聊 → `POST /chat`（**非流式 JSON**，走意图识别 + 记忆召回，不碰工具）
 * - `agent` 工作任务 → `POST /chat/with_agent`（**SSE 流式**，可触发审批/降级）
 *
 * ⚠️ 两者返回形态不同：闲聊一次拿到完整 JSON，工作任务逐段吐 SSE。
 * 状态机在 useChatStream 里据此分叉。
 */
export type ChatMode = "chat" | "agent";

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

/** 执行过程中的一步（工具调用 / 子任务结论）。后端在跑图时逐条推送。 */
export interface AgentStep {
  node: string;
  tool: string | null;
  title: string;
  status: string;
  detail: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  /** 这条回答由哪种模式产生（旧数据没有该字段，按工作任务处理） */
  mode?: ChatMode;
  meta?: MessageMeta;
  /** 流式被中断（用户切页/卸载），这一轮没有正常走完 */
  interrupted?: boolean;
  error?: string;
  /** 该助手消息上挂着一个待审批项 */
  approval?: ApprovalRequest;
  /** 执行过程（Agent 跑图时逐条推送，用于让用户看见"它在干什么"） */
  steps?: AgentStep[];
  /** 改写阶段的逐字缓冲（后端在模型生成时推，不是生成完再切片） */
  rewriteText?: string;
  /**
   * 因模型切换/重试被废弃的段落。
   *
   * ⚠️ 用户选的是"保留并标注"而不是清空：换候选时旧内容移到这里，
   * 界面上划掉并标注"上段因模型切换已废弃"——静默丢弃会让用户以为答案变了。
   */
  abandoned?: string[];
  /** 该消息对应的审批已失效 */
  approvalExpired?: boolean;
}
