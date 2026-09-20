import { useState } from "react";

import type { AgentStep } from "@/features/chat/types";

/** 与后端 `_common._LEDGER_BAD_STATUSES` 同源：这些状态代表这一步没拿到东西。 */
const BAD_STATUSES = new Set([
  "error",
  "empty_data",
  "budget_denied",
  "approval_denied",
]);

/** 短于此长度的详情不值得给"展开"按钮（反正两行能放下）。 */
const COLLAPSED_HINT_CHARS = 90;

function StepRow({ step, index }: { step: AgentStep; index: number }) {
  const [open, setOpen] = useState(false);
  const bad = BAD_STATUSES.has(step.status);
  const expandable = step.detail.length > COLLAPSED_HINT_CHARS;

  return (
    <li className="py-1 text-[12px]">
      <div className="flex gap-2">
        <span className="shrink-0 pt-0.5 text-fg-subtle">{index + 1}.</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="font-medium text-fg-muted">
              {/* react 的"直接给答案"那一轮没有工具名，标成"生成结论"更可读 */}
              {step.tool || step.title || (step.status === "final" ? "生成结论" : step.node)}
            </span>
            <span className={bad ? "text-danger-text" : "text-fg-subtle"}>
              {step.status}
            </span>
            {expandable && (
              <button
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
                className="text-[11px] text-accent-text transition-colors hover:underline"
              >
                {open ? "收起" : `展开（${step.detail.length} 字）`}
              </button>
            )}
          </div>

          {step.detail && (
            <div
              className={
                open
                  ? // 展开态：完整原文，自己滚，不撑破外层列表
                    "mt-1 max-h-96 overflow-y-auto rounded-md bg-surface-3 p-2 font-mono text-[11px] leading-relaxed break-words whitespace-pre-wrap text-fg-muted"
                  : // 收起态：只给两行预览，且不换行视图（避免短内容被拆成很多行）
                    "mt-0.5 line-clamp-2 text-[11px] break-words text-fg-subtle"
              }
            >
              {step.detail}
            </div>
          )}
        </div>
      </div>
    </li>
  );
}

/**
 * 执行过程（工具调用 / 子任务结论 / 改写逐字）。
 *
 * 后端在**图还在跑**的时候逐条推送，所以这块是"Agent 还在工作"的主要反馈。
 *
 * 两级折叠：整块可收起；每条的工具观测默认两行、可单独展开看全文
 * （后端最多推 5000 字符，够看原始检索片段）。
 *
 * `rewriteText` 是问题改写的逐字缓冲——它比第一个工具步骤更早出现，
 * 所以"还没有任何步骤、但已有改写内容"是正常的中间态，不能因为没有步骤
 * 就整块不渲染。
 */
export function RunSteps({
  steps,
  running,
  rewriteText,
}: {
  steps: AgentStep[];
  running: boolean;
  /** 改写阶段的逐字内容（可能先于任何步骤出现） */
  rewriteText?: string;
}) {
  const [open, setOpen] = useState(true);

  if (steps.length === 0 && !rewriteText) return null;
  const last = steps[steps.length - 1];

  // 只有改写、还没有步骤时不能说"0 步"——那是假信息
  const label =
    steps.length > 0
      ? `${running ? "执行中" : "执行过程"} · ${steps.length} 步`
      : running
        ? "改写中"
        : "问题改写";

  return (
    <div className="mb-3 overflow-hidden rounded-lg border border-line bg-surface-2/60">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-surface-2"
      >
        {running ? (
          <span className="agent-dots text-accent-text" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
        ) : null}
        <span className="shrink-0 text-[12px] font-medium text-fg-muted">
          {label}
        </span>
        <span className="min-w-0 flex-1 truncate text-[11px] text-fg-subtle">
          {last?.tool || last?.title || last?.node || ""}
        </span>
        <svg
          viewBox="0 0 24 24"
          className={`h-3.5 w-3.5 shrink-0 text-fg-subtle transition-transform ${open ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={2.4}
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && (
        <ol className="border-t border-line px-3 py-2">
          {/* 改写逐字：排在所有步骤之前——它就是链路里最早发生的一步 */}
          {rewriteText && (
            <li className="py-1 text-[12px]">
              <div className="flex gap-2">
                <span className="shrink-0 pt-0.5 text-fg-subtle">·</span>
                <div className="min-w-0 flex-1">
                  <div className="font-medium text-fg-muted">改写后的问题</div>
                  <div className="mt-0.5 break-words whitespace-pre-wrap text-[11px] text-fg-muted">
                    {rewriteText}
                    {running && !steps.length && (
                      <span className="agent-caret" aria-hidden="true" />
                    )}
                  </div>
                </div>
              </div>
            </li>
          )}

          {steps.map((s, i) => (
            <StepRow key={i} step={s} index={i} />
          ))}
        </ol>
      )}
    </div>
  );
}
