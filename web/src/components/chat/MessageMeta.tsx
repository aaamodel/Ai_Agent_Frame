import type { MessageMeta as Meta } from "@/features/chat/types";

/**
 * 消息脚注：执行步数 / 链路追踪 ID / 会话 ID。
 *
 * 闲聊模式（/chat）后端不产出步数，`stepsExecuted` 会是 0——
 * 这时省略"执行 N 步"，避免出现"执行 0 步"这种误导性表述。
 */
export function MessageMeta({ meta }: { meta: Meta }) {
  const parts = [
    meta.stepsExecuted > 0 ? `执行 ${meta.stepsExecuted} 步` : null,
    `trace ${meta.traceId || "—"}`,
    `session ${meta.sessionId || "—"}`,
  ].filter((p): p is string => p !== null);

  return (
    <div className="mt-3 border-t border-line pt-2 text-xs text-fg-subtle">
      {parts.join(" · ")}
    </div>
  );
}
