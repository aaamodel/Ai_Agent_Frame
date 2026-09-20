import { Link, NavLink, useSearchParams } from "react-router-dom";

import type { KbCollection } from "@/api/types";
import type { Session } from "@/features/chat/sessions";

/** 今天显示 HH:MM，昨天显示「昨天」，更早显示 M-D。 */
function formatTime(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) {
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) return "昨天";
  return `${d.getMonth() + 1}-${String(d.getDate()).padStart(2, "0")}`;
}

/** 列表项的摘要行：最后一条消息的文本；挂起审批时如实说明。 */
function summary(s: Session): string {
  const last = s.messages[s.messages.length - 1];
  if (!last) return "还没有消息";
  const text = (last.text ?? "").replace(/\s+/g, " ").trim();
  if (!text) return last.approval ? "等待你审批" : "…";
  const chars = [...text];
  return chars.length <= 40 ? text : chars.slice(0, 40).join("") + "…";
}

const ITEM_BASE =
  "block rounded-lg px-2.5 py-2 transition-colors max-[899px]:px-2";

/** 中间面板下半栏（对话区）：会话列表。 */
export function SessionListPane({
  sessions,
  onDelete,
}: {
  sessions: Session[];
  /** 删除某会话（含后端数据）。不传则不渲染删除按钮。 */
  onDelete?: (session: Session) => void;
}) {
  // ⚠️ 用查询串判断选中项，而不是 NavLink 的 isActive：
  //   NavLink 只比对 pathname，所有 /?session=xxx 都会命中 "/"，导致整列同时高亮。
  const [searchParams] = useSearchParams();
  const current = searchParams.get("session") ?? "";

  if (sessions.length === 0) {
    return (
      <div className="px-2 py-6 text-center text-xs text-fg-subtle">
        还没有会话，点「新会话」开始
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-0.5">
      {sessions.map((s) => {
        const active = s.id === current;
        return (
          /**
           * ⚠️ 删除按钮**不能**放进 `<Link>` 里：嵌在链接内部的按钮被点击时，
           *    事件会冒泡到链接上，删完还会顺手跳转到那个刚被删掉的会话。
           *    所以这里是 flex 容器，链接与按钮互为兄弟节点。
           */
          <div key={s.id} className="flex items-center gap-0.5">
            <Link
              to={`/?session=${s.id}`}
              className={[
                ITEM_BASE,
                "min-w-0 flex-1",
                active ? "bg-accent-soft" : "hover:bg-surface-2",
              ].join(" ")}
            >
              <div
                className={`truncate text-[13px] ${
                  active ? "font-semibold text-fg" : "font-medium text-fg-muted"
                }`}
              >
                {s.title}
              </div>
              <div className="mt-0.5 flex items-baseline gap-2">
                <span className="min-w-0 flex-1 truncate text-[11px] text-fg-subtle">
                  {summary(s)}
                </span>
                <span className="shrink-0 text-[11px] text-fg-subtle">
                  {formatTime(s.updatedAt)}
                </span>
              </div>
            </Link>

            {onDelete && (
              <button
                onClick={() => onDelete(s)}
                title={`删除会话「${s.title}」`}
                aria-label={`删除会话 ${s.title}`}
                className="grid h-7 w-7 shrink-0 place-items-center rounded-lg text-fg-subtle transition-colors hover:bg-surface-2 hover:text-danger-text"
              >
                <svg
                  viewBox="0 0 24 24"
                  className="h-3.5 w-3.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1.8}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v5M14 11v5" />
                </svg>
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** 中间面板下半栏（文档 / 知识库区）：集合列表。 */
export function CollectionListPane({
  items,
  base,
  emptyHint,
}: {
  items: KbCollection[];
  base: string;
  emptyHint: string;
}) {
  if (items.length === 0) {
    return (
      <div className="px-2 py-6 text-center text-xs text-fg-subtle">
        {emptyHint}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-0.5">
      {items.map((c) => (
        <NavLink
          key={c.name}
          to={`${base}/${encodeURIComponent(c.name)}`}
          className={({ isActive }) =>
            [
              ITEM_BASE,
              isActive ? "bg-accent-soft" : "hover:bg-surface-2",
            ].join(" ")
          }
        >
          {({ isActive }) => (
            <>
              <div
                className={`truncate text-[13px] ${
                  isActive ? "font-semibold text-fg" : "font-medium text-fg-muted"
                }`}
              >
                {c.name}
              </div>
              <div className="mt-0.5 flex items-baseline gap-2">
                <span className="min-w-0 flex-1 truncate text-[11px] text-fg-subtle">
                  {c.description || "（无描述）"}
                </span>
                <span className="shrink-0 text-[11px] text-fg-subtle">
                  {c.document_count} 文件
                </span>
              </div>
            </>
          )}
        </NavLink>
      ))}
    </div>
  );
}
