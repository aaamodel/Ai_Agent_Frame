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
    <div className="my-3 rounded-lg border border-warn-border border-l-4 border-l-warn-border bg-warn-bg p-3">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-warn-text">
        <span>⚠ 需要你审批</span>
        <span className="rounded-full border border-warn-border px-1.5 text-[11px] font-normal">
          {request.approvals.length} 项
        </span>
      </div>
      <ul className="mb-2 space-y-1 text-sm text-warn-text">
        {request.approvals.map((a, i) => (
          <li
            key={i}
            className="rounded-md bg-surface-0/50 px-2 py-1 font-mono text-xs break-all"
          >
            {describe(a)}
          </li>
        ))}
      </ul>
      <input
        className="mb-2 w-full rounded-lg border border-warn-border/60 bg-surface-0/40 px-2 py-1.5 text-sm text-fg placeholder:text-fg-subtle focus:border-warn-border focus:outline-none"
        placeholder="备注（拒绝时会回注给模型）"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        disabled={disabled}
      />
      <div className="flex gap-2">
        <button
          className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-50"
          disabled={disabled}
          onClick={() => onDecide({ approved: true, comment })}
        >
          批准
        </button>
        <button
          className="rounded-lg border border-line bg-transparent px-3 py-1.5 text-sm text-fg-muted transition-colors hover:bg-surface-2 hover:text-fg disabled:opacity-50"
          disabled={disabled}
          onClick={() => onDecide({ approved: false, comment })}
        >
          拒绝
        </button>
      </div>
    </div>
  );
}
