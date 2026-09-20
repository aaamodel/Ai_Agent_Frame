import type { ReactElement } from "react";

import { useNavigate, NavLink } from "react-router-dom";

import type { PendingApproval } from "@/api/types";
import { createSession, saveSession } from "@/features/chat/sessions";

/** 导航项图标（stroke 风格，随 currentColor 变色）。 */
const ICONS: Record<string, ReactElement> = {
  "/": (
    <path d="M20 15a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2z" />
  ),
  "/documents": (
    <>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
    </>
  ),
  "/knowledgebase": (
    <>
      <circle cx="6" cy="7" r="2.4" />
      <circle cx="18" cy="7" r="2.4" />
      <circle cx="12" cy="18" r="2.4" />
      <path d="M8.2 8.6 10.6 15.8M15.8 8.6 13.4 15.8M8.4 7h7.2" />
    </>
  ),
};

const NAV = [
  { to: "/", label: "对话" },
  { to: "/documents", label: "文档" },
  { to: "/knowledgebase", label: "知识库" },
];

/**
 * 最左侧窄导航栏：Logo + 三个区导航 + 待审批角标 + 底部「新会话」主按钮。
 *
 * ⚠️ 待审批角标跨区常驻，**不属于任何一个区** —— 它是挂起项在别的区
 *    唯一能被看见的地方（内联审批卡片只在对话区可见）。所以它放在这里，
 *    而不是放进中间那个会随区切换的上下文面板。
 *
 * 窄屏（<900px）只留图标：文字由 `max-[899px]:hidden` 收掉。
 */
export function Sidebar({
  pendingCount,
  pendingItems,
  backend,
}: {
  pendingCount: number;
  pendingItems: PendingApproval[];
  backend?: string;
}) {
  const navigate = useNavigate();

  function newSession() {
    const s = createSession();
    saveSession(s);
    void navigate(`/?session=${s.id}`);
  }

  return (
    <aside className="flex w-[220px] shrink-0 flex-col border-r border-line bg-surface-1 max-[899px]:w-14">
      {/* Logo / 产品名 */}
      <div className="flex h-14 shrink-0 items-center gap-2.5 px-4 max-[899px]:justify-center max-[899px]:px-0">
        <div className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-accent text-[13px] font-bold text-white">
          A
        </div>
        <span className="truncate text-[15px] font-semibold tracking-tight text-fg max-[899px]:hidden">
          Agent 控制台
        </span>
      </div>

      {/* 导航项 */}
      <nav className="flex flex-col gap-1 px-2 py-2">
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.to === "/"}
            title={n.label}
            className={({ isActive }) =>
              [
                "relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                "max-[899px]:justify-center max-[899px]:gap-0 max-[899px]:px-0",
                // 选中态：左侧色条 + 背景 + 加粗
                isActive
                  ? "bg-accent-soft font-semibold text-fg before:absolute before:left-0 before:top-1/2 before:h-5 before:w-[3px] before:-translate-y-1/2 before:rounded-full before:bg-accent"
                  : "text-fg-muted hover:bg-surface-2 hover:text-fg",
              ].join(" ")
            }
          >
            <svg
              viewBox="0 0 24 24"
              className="h-[18px] w-[18px] shrink-0"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.8}
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              {ICONS[n.to]}
            </svg>
            <span className="max-[899px]:hidden">{n.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* 撑开中间，把下面两块压到底部 */}
      <div className="min-h-0 flex-1" />

      {/* 待审批角标（跨区常驻） */}
      {pendingCount > 0 && (
        <div className="px-2 pb-2">
          <div className="rounded-lg border border-warn-border bg-warn-bg px-2.5 py-2 max-[899px]:px-0 max-[899px]:text-center">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-warn-text max-[899px]:justify-center">
              <span className="max-[899px]:hidden">⚠ 待审批</span>
              <span className="rounded-full bg-warn-text px-1.5 text-[11px] leading-5 text-surface-0">
                {pendingCount}
              </span>
            </div>
            <div className="mt-0.5 truncate text-[11px] text-warn-text/80 max-[899px]:hidden">
              {/* ⚠️ 后端字段是 user_input；工具名在嵌套的 approvals[] 里 */}
              {pendingItems[0]?.user_input ??
                pendingItems[0]?.approvals?.[0]?.tool_name ??
                "—"}
            </div>
            {backend === "memory" && (
              <div className="mt-1 text-[11px] text-warn-text max-[899px]:hidden">
                当前为内存模式，后端重启后将失效
              </div>
            )}
          </div>
        </div>
      )}

      {/* 新会话（固定在导航栏底部） */}
      <div className="shrink-0 border-t border-line p-2">
        <button
          onClick={newSession}
          title="新会话"
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-accent-hover max-[899px]:gap-0 max-[899px]:px-0"
        >
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4 shrink-0"
            fill="none"
            stroke="currentColor"
            strokeWidth={2.2}
            strokeLinecap="round"
            aria-hidden="true"
          >
            <path d="M12 5v14M5 12h14" />
          </svg>
          <span className="max-[899px]:hidden">新会话</span>
        </button>
      </div>
    </aside>
  );
}
