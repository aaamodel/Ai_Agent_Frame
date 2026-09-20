import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiGet, apiPostJson, API_BASE } from "./api";

afterEach(() => vi.restoreAllMocks());

describe("api 封装", () => {
  it("API_BASE 以 /api/v1 开头", () => {
    expect(API_BASE).toBe("/api/v1");
  });

  it("GET 拼出正确 URL 并解析 JSON", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: 1 }), { status: 200 }),
    );
    await expect(apiGet("/documents")).resolves.toEqual({ ok: 1 });
    expect(spy).toHaveBeenCalledWith("/api/v1/documents", expect.anything());
  });

  it("非 2xx 抛 ApiError 并带上 detail", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "集合不存在或为空" }), {
        status: 404,
      }),
    );
    // 用 try/catch 而非 .catch(e => e)：后者会把结果推成 unknown，属性访问通不过类型检查
    let err: unknown = undefined;
    try {
      await apiGet("/vector/collections/nope/files");
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(404);
    expect((err as ApiError).detail).toBe("集合不存在或为空");
  });

  it("响应不是 JSON 时不崩，detail 退回状态文本", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("<html>500</html>", { status: 500 }),
    );
    let err: unknown = undefined;
    try {
      await apiGet("/documents");
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(500);
    expect((err as ApiError).detail.length).toBeGreaterThan(0);
  });

  it("POST JSON 带上 Content-Type 与序列化后的 body", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({}), { status: 200 }),
    );
    await apiPostJson("/agent/runs/x/approval", {
      approved: true,
      comment: "ok",
    });
    const [, init] = spy.mock.calls[0];
    expect(init?.method).toBe("POST");
    expect((init?.headers as Record<string, string>)["Content-Type"]).toContain(
      "application/json",
    );
    expect(init?.body).toBe('{"approved":true,"comment":"ok"}');
  });
});
