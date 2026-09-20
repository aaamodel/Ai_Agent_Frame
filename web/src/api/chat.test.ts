import { afterEach, describe, expect, it, vi } from "vitest";

import { deleteChatSession } from "./chat";

afterEach(() => vi.restoreAllMocks());

describe("deleteChatSession", () => {
  it("DELETE 到 /chat/sessions/{id}，并对 id 做 URL 编码", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: "a b",
          memory: { short_term: "ok", long_term: "ok" },
          removed_runs: 2,
        }),
        { status: 200 },
      ),
    );

    const result = await deleteChatSession("a b");

    const [url, init] = spy.mock.calls[0] as [string, RequestInit];
    expect(String(url)).toBe(
      `/api/v1/chat/sessions/${encodeURIComponent("a b")}`,
    );
    expect(init.method).toBe("DELETE");
    expect(result.memory.short_term).toBe("ok");
    expect(result.removed_runs).toBe(2);
  });

  it("后端返回非 2xx 时抛错（调用方据此保留本地会话）", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "清空短期记忆失败: redis 不可用" }), {
        status: 500,
      }),
    );

    await expect(deleteChatSession("s1")).rejects.toThrow(/redis 不可用/);
  });
});
