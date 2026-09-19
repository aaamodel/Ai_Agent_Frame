import { useEffect, useState } from "react";

import { listSessions, subscribeSessions, type Session } from "./sessions";

/**
 * 左栏下半栏（对话区）的数据源：会话列表。
 *
 * 订阅存储变更而不是只在挂载时读一次——否则对话页新建会话后侧栏不会刷新。
 */
export function useSessionList(): { sessions: Session[] } {
  const [sessions, setSessions] = useState<Session[]>(() => listSessions());

  useEffect(() => subscribeSessions(() => setSessions(listSessions())), []);

  return { sessions };
}
