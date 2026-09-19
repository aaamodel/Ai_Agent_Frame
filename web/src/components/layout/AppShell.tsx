import { NavLink, Outlet, useLocation } from "react-router-dom";

import type { KbCollection } from "@/api/types";
import { usePendingApprovals } from "@/features/approvals/usePendingApprovals";
import { usePendingCountInDocumentTitle } from "@/features/approvals/usePendingApprovals";
import type { Session } from "@/features/chat/sessions";
import { useSessionList } from "@/features/chat/useSessionList";
import { useKbCollectionList } from "@/features/documents/useKbCollectionList";
import { useGraphCollectionList } from "@/features/knowledgebase/useGraphCollectionList";

import { Sidebar } from "./Sidebar";

/** 左栏下半栏（对话区）：会话列表。 */
function SessionListPane({ sessions }: { sessions: Session[] }) {
  if (sessions.length === 0) {
    return <div className="px-2 py-1 text-xs text-neutral-400">暂无会话</div>;
  }
  return (
    <>
      <div className="px-2 pb-1 text-xs text-neutral-400">会话</div>
      {sessions.map((s) => (
        <NavLink
          key={s.id}
          to={`/?session=${s.id}`}
          className="block truncate rounded px-2 py-1 text-sm text-neutral-600 hover:bg-neutral-100"
        >
          {s.title}
        </NavLink>
      ))}
    </>
  );
}

/** 左栏下半栏（文档 / 知识库区）：集合列表。 */
function CollectionListPane({
  title,
  items,
  base,
}: {
  title: string;
  items: KbCollection[];
  base: string;
}) {
  if (items.length === 0) {
    return <div className="px-2 py-1 text-xs text-neutral-400">暂无集合</div>;
  }
  return (
    <>
      <div className="px-2 pb-1 text-xs text-neutral-400">{title}</div>
      {items.map((c) => (
        <NavLink
          key={c.name}
          to={`${base}/${encodeURIComponent(c.name)}`}
          className="block truncate rounded px-2 py-1 text-sm text-neutral-600 hover:bg-neutral-100"
        >
          {c.name}
        </NavLink>
      ))}
    </>
  );
}

/**
 * 应用外壳：左栏（导航 + 随区切换的下半栏 + 底部待审批角标）+ 主区。
 *
 * ⚠️ 待审批角标跨区常驻，**不属于任何一个区** —— 它是挂起项在别的区
 *    唯一能被看见的地方（内联卡片只在对话区可见）。
 */
export function AppShell() {
  const { count, items, backend } = usePendingApprovals();
  usePendingCountInDocumentTitle(count);
  const { pathname } = useLocation();

  const { sessions } = useSessionList();
  const kbCollections = useKbCollectionList();
  const graphCollections = useGraphCollectionList();

  const lower = pathname.startsWith("/documents") ? (
    <CollectionListPane
      title="RAG 集合"
      items={kbCollections.data ?? []}
      base="/documents"
    />
  ) : pathname.startsWith("/knowledgebase") ? (
    <CollectionListPane
      title="图谱集合"
      items={graphCollections.data ?? []}
      base="/knowledgebase"
    />
  ) : (
    <SessionListPane sessions={sessions} />
  );

  return (
    <div className="flex h-full">
      <Sidebar
        pendingCount={count}
        pendingItems={items}
        backend={backend}
        lower={lower}
      />
      <main className="min-w-0 flex-1">
        <Outlet />
      </main>
    </div>
  );
}
