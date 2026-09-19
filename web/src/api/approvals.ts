import { apiGet, apiPostStream } from "@/lib/api";
import type {
  ApprovalDecision,
  PendingApprovalsResponse,
  RunStatus,
} from "./types";

export async function listPendingApprovals(): Promise<PendingApprovalsResponse> {
  return apiGet<PendingApprovalsResponse>("/agent/approvals/pending");
}

export async function getRunStatus(runId: string): Promise<RunStatus> {
  return apiGet<RunStatus>(`/agent/runs/${encodeURIComponent(runId)}`);
}

/** 提交审批决定并拿到续流（SSE）。 */
export async function apiResumeRun(
  runId: string,
  decision: ApprovalDecision,
  signal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  return apiPostStream(
    `/agent/runs/${encodeURIComponent(runId)}/approval`,
    decision,
    signal,
  );
}
