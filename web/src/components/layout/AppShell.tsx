import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { Outlet, useLocation, useNavigate, useSearchParams } from "react-router-dom";

import { deleteChatSession } from "@/api/chat";
import { usePendingApprovals } from "@/features/approvals/usePendingApprovals";
import { usePendingCountInDocumentTitle } from "@/features/approvals/usePendingApprovals";
import { createSession, deleteSession, saveSession } from "@/features/chat/sessions";
import type { Session } from "@/features/chat/sessions";
import { useSessionList } from "@/features/chat/useSessionList";
import { useKbCollectionList } from "@/features/documents/useKbCollectionList";
import { useGraphCollectionList } from "@/features/knowledgebase/useGraphCollectionList";

import { ContextPanel } from "./ContextPanel";
import { CollectionListPane, SessionListPane } from "./ContextPanes";
import { Sidebar } from "./Sidebar";

const NARROW_QUERY = "(max-width: 899px)";

/** 窄屏（<900px）。中间面板在窄屏下默认收起。 */
function useIsNarrow(): boolean {
  // ⚠️ jsdom（测试环境）没有实现 window.matchMedia，直接调用会抛错并让整个外壳崩掉。
  const [narrow, setNarrow] = useState(() => matchMediaSafe()?.matches ?? false);
  useEffect(() => {
    const mq = matchMediaSafe();
    if (!mq) return;
    const onChange = (e: MediaQueryListEvent) => setNarrow(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

function matchMediaSafe(): MediaQueryList | null {
  if (typeof window === "undefined") return null;
  if (typeof window.matchMedia !== "function") return null;
  return window.matchMedia(NARROW_QUERY);
}

/**
 * 应用外壳：三栏。
 *
 *   左（220px 导航） + 中（264px 上下文，可折叠） + 右（主区，自适应）
 *
 * 三栏各自独立滚动，整页不出现滚动条（body overflow:hidden + 各栏 overflow-y-auto）。
 *
 * ⚠️ 待审批角标放在**左栏**而非中间面板 —— 中间面板的内容会随区切换，
 *    而角标必须跨区常驻：它是挂起项在别的区唯一能被看见的地方。
 */
export function AppShell() {
  const { count, items, backend } = usePendingApprovals();
  usePendingCountInDocumentTitle(count);
  const { pathname } = useLocation();
  const navigate = useNavigate();
  // 判断被删的会话是不是"当前正在看的那个"（会话由 ?session= 标识）
  const [searchParams] = useSearchParams();

  const { sessions } = useSessionList();
  const kbCollections = useKbCollectionList();
  const graphCollections = useGraphCollectionList();

  // 默认跟随屏宽；用户手动折叠/展开后以手动为准（manual 为 null 时交给 isNarrow）
  const isNarrow = useIsNarrow();
  const [manualCollapsed, setManualCollapsed] = useState<boolean | null>(null);
  const collapsed = manualCollapsed ?? isNarrow;

  function newSession() {
    const s = createSession();
    saveSession(s);
    void navigate(`/?session=${s.id}`);
  }

  /**
   * 删除会话：**先确认 → 再删后端 → 最后删本地**。
   *
   * 顺序是关键：后端失败时**不删本地**。反过来的话本地记录没了、后端记忆还在，
   * 就成了"删了还在"——下一轮召回仍会命中旧记录，用户却已经看不到那条会话，
   * 想重删都没有入口。
   */
  async function handleDeleteSession(session: Session) {
    const confirmed = window.confirm(
      `确认删除会话「${session.title}」？\n\n` +
        "这会同时删除后端保存的该会话记忆（短期 + 长期）与检查点，且不可撤销。",
    );
    if (!confirmed) return;

    try {
      const result = await deleteChatSession(session.id);
      // 后端逐项回传状态：只要有一项失败就如实说明，不谎报"已删干净"
      const failed = Object.entries(result.memory ?? {}).filter(
        ([, value]) => value !== "ok",
      );
      if (failed.length > 0) {
        window.alert(
          "后端只清掉了一部分，本地会话已保留：\n" +
            failed.map(([key, value]) => `- ${key}: ${value}`).join("\n"),
        );
        return;
      }
    } catch (e) {
      window.alert(
        `后端删除失败：${e instanceof Error ? e.message : String(e)}\n\n` +
          "本地会话未删除，请重试。",
      );
      return;
    }

    deleteSession(session.id);

    // 删掉的正是当前会话：换一个新会话，否则界面会停在一个已不存在的会话上
    if (searchParams.get("session") === session.id) {
      const fresh = createSession();
      saveSession(fresh);
      void navigate(`/?session=${fresh.id}`);
    }
  }

  let panelTitle = "会话";
  let panelNew: (() => void) | undefined = newSession;
  let panelBody: ReactNode;

  if (pathname.startsWith("/documents")) {
    panelTitle = "RAG 集合";
    panelNew = undefined;
    panelBody = (
      <CollectionListPane
        items={kbCollections.data ?? []}
        base="/documents"
        emptyHint="还没有 RAG 集合"
      />
    );
  } else if (pathname.startsWith("/knowledgebase")) {
    panelTitle = "图谱集合";
    panelNew = undefined;
    panelBody = (
      <CollectionListPane
        items={graphCollections.data ?? []}
        base="/knowledgebase"
        emptyHint="还没有图谱集合"
      />
    );
  } else {
    panelBody = (
      <SessionListPane sessions={sessions} onDelete={handleDeleteSession} />
    );
  }

  return (
    <div className="flex h-full overflow-hidden bg-surface-0 text-fg">
      <Sidebar
        pendingCount={count}
        pendingItems={items}
        backend={backend}
      />
      <ContextPanel
        title={panelTitle}
        collapsed={collapsed}
        onToggle={() => setManualCollapsed(!collapsed)}
        onNew={panelNew}
        newLabel="新会话"
      >
        {panelBody}
      </ContextPanel>
      {/* 主区：自身不滚动，滚动交给各页面内部，避免整页出现滚动条 */}
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
