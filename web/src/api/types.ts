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
  /**
   * ⚠️ 图谱侧的字段名是 `chunks_count`（多一个 s），
   * 与 RAG 侧的 `chunk_count` **不同名**（见 app/api/routes/kownledgebase.py:98
   * 与 document.py:120）。两个都留着，读取时按顺序回退。
   */
  chunks_count?: number;
  chunk_count?: number;
}

export interface GraphCollection {
  name: string;
  legacy: boolean;
  description: string | null;
  document_count: number;
  files: GraphFile[];
}

/**
 * GET /agent/approvals/pending 的单条记录。
 *
 * ⚠️ 字段名按后端**实际返回**（app/core/agent/graph/runner.py:255-278）：
 * 原始问题是 `user_input`（不是 query），工具名与参数在**嵌套的** `approvals[]` 里
 * （`tool_name` / `arguments_preview`），并不在顶层。
 */
export interface PendingApproval {
  run_id: string;
  session_id: string;
  trace_id?: string;
  user_input?: string;
  mode?: string;
  paused_at?: string;
  /** 嵌套的审批内容：{tool_name, arguments_preview, subtask_id, ...} */
  approvals?: Array<{
    tool_name?: string;
    arguments_preview?: string;
    subtask_id?: string;
  }>;
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
