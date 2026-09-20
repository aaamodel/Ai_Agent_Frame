import { useState } from "react";

import type { ChatMode } from "@/features/chat/types";

const MODE_LABEL: Record<ChatMode, string> = {
  chat: "对话",
  agent: "工作任务",
};

const MODE_DESC: Record<ChatMode, string> = {
  chat: "闲聊：不调用工具与知识库",
  agent: "调用工具与知识库，可能需要审批",
};

const ORDER: ChatMode[] = ["chat", "agent"];

/** 模式选择器（豆包式：挂在输入框左下角，菜单向上弹）。 */
function ModePicker({
  mode,
  onChange,
  disabled,
}: {
  mode: ChatMode;
  onChange: (m: ChatMode) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        title="切换对话模式"
        aria-label="切换对话模式"
        aria-expanded={open}
        className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-medium text-fg-muted transition-colors hover:bg-surface-2 hover:text-fg disabled:opacity-50"
      >
        {MODE_LABEL[mode]}
        <svg
          viewBox="0 0 24 24"
          className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`}
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
        <>
          {/* 透明遮罩：点外面就收起，不必监听全局 click */}
          <div
            className="fixed inset-0 z-30"
            onClick={() => setOpen(false)}
            aria-hidden="true"
          />
          <div className="absolute bottom-full left-0 z-40 mb-2 w-60 rounded-lg border border-line bg-surface-2 p-1 shadow-xl">
            {ORDER.map((m) => {
              const active = m === mode;
              return (
                <button
                  key={m}
                  onClick={() => {
                    onChange(m);
                    setOpen(false);
                  }}
                  className={[
                    "flex w-full flex-col items-start gap-0.5 rounded-md px-2.5 py-1.5 text-left transition-colors",
                    active
                      ? "bg-accent-soft"
                      : "hover:bg-surface-3",
                  ].join(" ")}
                >
                  <span
                    className={`text-[13px] ${
                      active ? "font-semibold text-fg" : "text-fg-muted"
                    }`}
                  >
                    {MODE_LABEL[m]}
                  </span>
                  <span className="text-[11px] text-fg-subtle">
                    {MODE_DESC[m]}
                  </span>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

export function Composer({
  disabled,
  isStreaming,
  onStop,
  onSend,
  mode,
  onModeChange,
}: {
  disabled: boolean;
  isStreaming: boolean;
  onStop: () => void;
  onSend: (text: string) => void;
  mode: ChatMode;
  onModeChange: (m: ChatMode) => void;
}) {
  const [text, setText] = useState("");

  function submit() {
    const t = text.trim();
    if (!t || disabled) return;
    setText("");
    onSend(t);
  }

  return (
    <div className="shrink-0 border-t border-line bg-surface-1 px-4 py-3">
      {/* 运行中的状态条：明确告诉用户"还在跑"，并给出可中断的出口 */}
      {isStreaming && (
        <div className="mx-auto mb-2 flex w-full max-w-3xl items-center gap-2 rounded-lg border border-accent-ring bg-accent-soft px-3 py-2 text-xs text-accent-text">
          <span className="agent-dots" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="min-w-0 flex-1">
            {mode === "chat" ? "正在思考…" : "Agent 正在运行…"}
          </span>
          <button
            onClick={onStop}
            className="shrink-0 rounded-md border border-accent-ring px-2 py-0.5 font-medium transition-colors hover:bg-accent hover:text-white"
          >
            停止
          </button>
        </div>
      )}

      <div className="mx-auto w-full max-w-3xl rounded-lg border border-line bg-surface-3 px-3 py-2 transition-colors focus-within:border-accent-ring">
        <textarea
          className="max-h-40 w-full resize-none bg-transparent text-sm text-fg outline-none placeholder:text-fg-subtle disabled:opacity-50"
          rows={1}
          placeholder={mode === "chat" ? "聊点什么…" : "问点什么…"}
          value={text}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            // Enter 发送，Shift+Enter 换行
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <div className="mt-1 flex items-center gap-2">
          {/* 模式选择器（豆包式，左下角） */}
          <ModePicker
            mode={mode}
            onChange={onModeChange}
            disabled={disabled}
          />
          <span className="min-w-0 flex-1 truncate text-[11px] text-fg-subtle">
            Enter 发送 · Shift+Enter 换行
          </span>
          <button
            className="shrink-0 rounded-lg bg-accent px-3.5 py-1 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
            onClick={submit}
            disabled={disabled}
          >
            发送
          </button>
        </div>
      </div>
    </div>
  );
}
