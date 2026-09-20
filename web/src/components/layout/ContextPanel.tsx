import type { ReactNode } from "react";

/**
 * 中间上下文面板：标题区 + 随区切换的列表 + 折叠按钮。
 *
 * 内容由 AppShell 按当前区注入（对话 → 会话列表；文档 → RAG 集合；知识库 → 图谱集合），
 * 面板本身不关心数据是哪种。
 *
 * 收起后只留一条 40px 窄边（含展开按钮），把宽度让给右侧主区。
 */
export function ContextPanel({
  title,
  collapsed,
  onToggle,
  onNew,
  newLabel,
  children,
}: {
  title: string;
  collapsed: boolean;
  onToggle: () => void;
  /** 标题区右侧的「新建」动作；不传则不渲染该按钮 */
  onNew?: () => void;
  newLabel?: string;
  children: ReactNode;
}) {
  if (collapsed) {
    return (
      <div className="flex w-10 shrink-0 flex-col items-center border-r border-line bg-surface-1 py-2">
        <button
          onClick={onToggle}
          title={`展开${title}面板`}
          aria-label={`展开${title}面板`}
          className="grid h-8 w-8 place-items-center rounded-lg text-fg-subtle transition-colors hover:bg-surface-2 hover:text-fg"
        >
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M9 6l6 6-6 6" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <aside className="flex w-[264px] shrink-0 flex-col border-r border-line bg-surface-1">
      {/* 标题区：与左栏 Logo、右栏标题栏同为 56px 高，三栏顶部对齐 */}
      <div className="flex h-14 shrink-0 items-center gap-1 border-b border-line px-3">
        <span className="min-w-0 flex-1 truncate text-[13px] font-semibold uppercase tracking-wide text-fg-muted">
          {title}
        </span>
        {onNew && (
          <button
            onClick={onNew}
            title={newLabel}
            aria-label={newLabel}
            className="grid h-7 w-7 shrink-0 place-items-center rounded-lg text-fg-subtle transition-colors hover:bg-surface-2 hover:text-fg"
          >
            <svg
              viewBox="0 0 24 24"
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              strokeWidth={2.2}
              strokeLinecap="round"
              aria-hidden="true"
            >
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
        )}
        {/* 折叠按钮 */}
        <button
          onClick={onToggle}
          title="收起面板"
          aria-label="收起面板"
          className="grid h-7 w-7 shrink-0 place-items-center rounded-lg text-fg-subtle transition-colors hover:bg-surface-2 hover:text-fg"
        >
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M15 6l-6 6 6 6" />
          </svg>
        </button>
      </div>

      {/* 列表：独立滚动 */}
      <div className="min-h-0 flex-1 overflow-y-auto p-2">{children}</div>
    </aside>
  );
}
