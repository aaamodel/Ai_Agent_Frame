import type { MessageMeta as Meta } from "@/features/chat/types";

/** 消息脚注：执行步数 / 链路追踪 ID / 会话 ID。 */
export function MessageMeta({ meta }: { meta: Meta }) {
  return (
    <div className="mt-2 border-t border-neutral-100 pt-1 text-xs text-neutral-400">
      执行 {meta.stepsExecuted} 步 · trace {meta.traceId || "—"} · session{" "}
      {meta.sessionId || "—"}
    </div>
  );
}
