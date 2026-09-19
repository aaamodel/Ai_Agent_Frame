/** 后端返回结构的前端映射。字段名与后端保持一字不差。 */

export interface KbCollection {
  name: string;
  description: string | null;
  document_count: number;
}

export interface KbFile {
  filename: string;
  chunk_count?: number;
  /** 后端在不同接口里用的字段名不完全一致，两个都留着 */
  chunks?: number;
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  collection_name: string;
  description?: string | null;
}

export interface GraphFile {
  filename: string;
  /** LightRAG 处理状态：已处理 / 处理中 / 失败 */
  status?: string;
  chunk_count?: number;
}

export interface GraphCollection {
  name: string;
  legacy: boolean;
  description: string | null;
  document_count: number;
  files: GraphFile[];
}

/** GET /agent/approvals/pending 的单条记录。 */
export interface PendingApproval {
  run_id: string;
  session_id: string;
  trace_id?: string;
  query?: string;
  mode?: string;
  tool_name?: string;
  args_preview?: string;
  subtask_id?: string;
  paused_at?: string;
}

export interface PendingApprovalsResponse {
  /** "redis" | "memory" —— memory 表示后端重启后挂起项失效 */
  backend: string;
  items: PendingApproval[];
}

/** GET /agent/runs/{run_id} 的快照。 */
export interface RunStatus {
  exists: boolean;
  paused?: boolean;
  next?: unknown;
  interrupts?: unknown;
  mode_used?: string;
  success?: boolean;
}

export interface ApprovalDecision {
  approved: boolean;
  comment: string;
}
