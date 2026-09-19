import type { ReactNode } from "react";

import { NavLink } from "react-router-dom";

import type { PendingApproval } from "@/api/types";

const NAV = [
  { to: "/", label: "💬 对话" },
  { to: "/documents", label: "📄 文档" },
  { to: "/knowledgebase", label: "🕸 知识库" },
];

export function Sidebar({
  pendingCount,
  pendingItems,
  backend,
  lower,
}: {
  pendingCount: number;
  pendingItems: PendingApproval[];
  backend?: string;
  /** 下半栏内容：随当前区切换（会话列表 / RAG 集合 / 图谱集合） */
  lower: ReactNode;
}) {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-neutral-200 bg-neutral-50">
      <nav className="p-2">
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.to === "/"}
            className={({ isActive }) =>
              `block rounded px-3 py-2 text-sm ${
                isActive
                  ? "bg-neutral-200 font-semibold"
                  : "text-neutral-600 hover:bg-neutral-100"
              }`
            }
          >
            {n.label}
          </NavLink>
        ))}
      </nav>

      <div className="min-h-0 flex-1 overflow-y-auto border-t border-dashed border-neutral-300 px-2 py-2">
        {lower}
      </div>

      {pendingCount > 0 && (
        <div className="border-t border-neutral-200 p-2">
          <div className="rounded border border-warn-border bg-warn-bg px-2 py-1.5 text-xs font-semibold text-warn-text">
            ⚠ 待审批
            <span className="ml-1 rounded-full bg-warn-text px-1.5 text-white">
              {pendingCount}
            </span>
          </div>
          <div className="mt-1 truncate text-[10px] text-neutral-400">
            {pendingItems[0]?.query ?? pendingItems[0]?.tool_name ?? "—"}
          </div>
          {backend === "memory" && (
            <div className="mt-1 text-[10px] text-warn-text">
              当前为内存模式，后端重启后将失效
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
