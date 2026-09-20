import type { ChatMessage } from "./types";

/**
 * 会话历史存在前端。
 *
 * ⚠️ 后端**没有**会话列表接口（全项目无 /sessions 之类端点），
 * 它只持有短期记忆。因此"我上次问了什么"只能由前端自己记。
 */
const STORAGE_KEY = "agent-console.sessions";

export interface Session {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
  /** 该会话当前挂起的审批 run_id（用于刷新后回查恢复） */
  pendingRunId?: string;
}

export function newSessionId(): string {
  return crypto.randomUUID();
}

export function createSession(): Session {
  return {
    id: newSessionId(),
    title: "新会话",
    updatedAt: Date.now(),
    messages: [],
  };
}

/** 取首条用户消息前 20 个码点作为标题。 */
export function deriveTitle(text: string): string {
  const t = (text ?? "").trim();
  if (!t) return "新会话";
  // 用扩展运算符按码点切，避免把 emoji / 代理对切成半个字符
  const chars = [...t];
  return chars.length <= 20 ? t : chars.slice(0, 20).join("") + "…";
}

function readAll(): Session[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Session[]) : [];
  } catch {
    // 存储被外部写坏时不崩——宁可丢历史，也不要整个页面打不开
    return [];
  }
}

function writeAll(sessions: Session[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {
    // 配额满 / 隐私模式：静默降级为"本次会话不持久化"
  }
}

export function listSessions(): Session[] {
  return readAll().sort((a, b) => b.updatedAt - a.updatedAt);
}

type SessionsListener = () => void;

const listeners = new Set<SessionsListener>();

/**
 * 订阅会话变更。
 *
 * 为什么需要：会话存储是模块级的 localStorage，而**侧栏**与**对话页**是两个
 * 互不相干的组件树分支。没有订阅的话，对话页新建会话后，侧栏的列表不会刷新，
 * 用户看不到刚产生的会话——功能上是坏的。
 */
export function subscribeSessions(fn: SessionsListener): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

function notifySessions(): void {
  for (const fn of listeners) fn();
}

export function getSession(id: string): Session | undefined {
  return readAll().find((s) => s.id === id);
}

export function saveSession(session: Session): void {
  const rest = readAll().filter((s) => s.id !== session.id);
  writeAll([...rest, { ...session, updatedAt: Date.now() }]);
  notifySessions();
}

export function deleteSession(id: string): void {
  writeAll(readAll().filter((s) => s.id !== id));
  notifySessions();
}
