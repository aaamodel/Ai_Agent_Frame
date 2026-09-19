import { useState } from "react";

import type { ApprovalDecision } from "@/api/types";
import type { ApprovalRequest } from "@/features/chat/types";

function describe(a: unknown): string {
  if (!a || typeof a !== "object") return String(a);
  const o = a as Record<string, unknown>;
  const tool = o.tool_name ? String(o.tool_name) : "未知工具";
  const args = o.args_preview ?? o.args;
  return args
    ? `${tool}：${typeof args === "string" ? args : JSON.stringify(args)}`
    : tool;
}

export function ApprovalCard({
  request,
  disabled,
  onDecide,
}: {
  request: ApprovalRequest;
  disabled: boolean;
  onDecide: (d: ApprovalDecision) => void;
}) {
  const [comment, setComment] = useState("");
  return (
    <div className="my-3 rounded-md border border-l-4 border-warn-border border-l-warn-border bg-warn-bg p-3">
      <div className="mb-1 font-semibold text-warn-text">⚠ 需要你审批</div>
      <ul className="mb-2 space-y-1 text-sm text-warn-text">
        {request.approvals.map((a, i) => (
          <li
            key={i}
            className="rounded bg-white/70 px-2 py-1 font-mono text-xs break-all"
          >
            {describe(a)}
          </li>
        ))}
      </ul>
      <input
        className="mb-2 w-full rounded border border-neutral-300 px-2 py-1 text-sm"
        placeholder="备注（拒绝时会回注给模型）"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        disabled={disabled}
      />
      <div className="flex gap-2">
        <button
          className="rounded bg-blue-600 px-3 py-1 text-sm text-white disabled:opacity-50"
          disabled={disabled}
          onClick={() => onDecide({ approved: true, comment })}
        >
          批准
        </button>
        <button
          className="rounded border border-neutral-300 bg-white px-3 py-1 text-sm disabled:opacity-50"
          disabled={disabled}
          onClick={() => onDecide({ approved: false, comment })}
        >
          拒绝
        </button>
      </div>
    </div>
  );
}
