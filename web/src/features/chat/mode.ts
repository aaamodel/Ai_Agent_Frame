import type { ChatMode } from "./types";

/**
 * 对话模式的本地持久化。
 *
 * 与会话历史一样存在 localStorage（后端没有会话/偏好类接口）。
 * 默认 `agent`：这是本控制台的主用途，升级后行为不变。
 */
const MODE_KEY = "agent-console.mode";
const DEFAULT_MODE: ChatMode = "agent";

export function readMode(): ChatMode {
  try {
    const raw = localStorage.getItem(MODE_KEY);
    return raw === "chat" || raw === "agent" ? raw : DEFAULT_MODE;
  } catch {
    // 隐私模式 / 存储被禁：退回默认，不让偏好读取影响功能
    return DEFAULT_MODE;
  }
}

export function saveMode(mode: ChatMode): void {
  try {
    localStorage.setItem(MODE_KEY, mode);
  } catch {
    // 配额满 / 隐私模式：静默降级为"本次会话内记住"
  }
}
