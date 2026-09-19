import { useState } from "react";

export function Composer({
  disabled,
  onSend,
}: {
  disabled: boolean;
  onSend: (text: string) => void;
}) {
  const [text, setText] = useState("");

  function submit() {
    const t = text.trim();
    if (!t || disabled) return;
    setText("");
    onSend(t);
  }

  return (
    <div className="border-t border-neutral-200 px-4 py-3">
      <div className="flex items-end gap-2 rounded-lg border border-neutral-300 px-3 py-2">
        <textarea
          className="max-h-40 flex-1 resize-none text-sm outline-none"
          rows={1}
          placeholder="问点什么…"
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
        <button
          className="rounded bg-blue-600 px-3 py-1 text-sm text-white disabled:opacity-40"
          onClick={submit}
          disabled={disabled}
        >
          发送
        </button>
      </div>
      <div className="mt-1 pl-1 text-xs text-neutral-400">
        Enter 发送 · Shift+Enter 换行
      </div>
    </div>
  );
}
