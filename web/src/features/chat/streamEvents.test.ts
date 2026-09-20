import { describe, expect, it } from "vitest";

import { toStreamEvent } from "./streamEvents";

describe("toStreamEvent", () => {
  it("含 content 的是正文片段", () => {
    expect(toStreamEvent({ content: "你", trace_id: "t1" })).toEqual({
      kind: "content",
      text: "你",
    });
  });

  it("awaiting_approval 优先于 done（同一个事件里两者都有）", () => {
    // 后端待审批事件同时带 done:true —— 必须判成审批，否则会把对话当成已结束
    const ev = toStreamEvent({
      awaiting_approval: true,
      run_id: "r1",
      approvals: [{ tool_name: "x" }],
      done: true,
      status: "awaiting_approval",
    });
    expect(ev?.kind).toBe("awaiting_approval");
    if (ev?.kind === "awaiting_approval") {
      expect(ev.runId).toBe("r1");
      expect(ev.approvals).toHaveLength(1);
    }
  });

  it("done 事件带上元数据", () => {
    const ev = toStreamEvent({
      done: true,
      status: "degraded",
      degraded: true,
      steps_executed: 4,
      trace_id: "t9",
      session_id: "s1",
    });
    expect(ev).toEqual({
      kind: "done",
      status: "degraded",
      degraded: true,
      stepsExecuted: 4,
      traceId: "t9",
      sessionId: "s1",
      runId: undefined,
    });
  });

  it("degraded 缺失时按 false 处理", () => {
    const ev = toStreamEvent({ done: true, status: "success" });
    if (ev?.kind === "done") expect(ev.degraded).toBe(false);
  });

  it("error 事件", () => {
    expect(toStreamEvent({ error: "后端炸了" })).toEqual({
      kind: "error",
      message: "后端炸了",
    });
  });

  it("认不出来的事件返回 null", () => {
    expect(toStreamEvent({ 无关字段: 1 })).toBeNull();
  });

  it("step 事件（执行过程）带上工具名与状态", () => {
    expect(
      toStreamEvent({
        step: {
          node: "execute",
          tool: "sales_sql_query",
          title: "查销售额",
          status: "ok",
          detail: "返回 12 行",
        },
      }),
    ).toEqual({
      kind: "step",
      node: "execute",
      tool: "sales_sql_query",
      title: "查销售额",
      status: "ok",
      detail: "返回 12 行",
    });
  });

  it("step 事件缺 status 时默认 ok，tool 为空时归 null", () => {
    const ev = toStreamEvent({ step: { node: "summarize", title: "汇总" } });
    expect(ev).toEqual({
      kind: "step",
      node: "summarize",
      tool: null,
      title: "汇总",
      status: "ok",
      detail: "",
    });
  });
});
