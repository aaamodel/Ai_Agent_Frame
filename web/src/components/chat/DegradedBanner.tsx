/**
 * 降级警示条。
 *
 * ⚠️ 只应在 `degraded === true` 时渲染。这个字段的语义是
 * "Agent 按现有信息给了诚实的部分答案"——把它藏进一行灰字，
 * 等于让后端那套"诚实部分答案"的机制白做。
 */
export function DegradedBanner() {
  return (
    <div className="mb-3 rounded-md border border-warn-border bg-warn-bg px-3 py-2 text-sm text-warn-text">
      <strong>⚠ 本次回答未取全信息</strong>
      <span className="ml-1">
        —— 部分依据未检索到，结论可能不完整，请勿直接作为决策依据
      </span>
    </div>
  );
}
