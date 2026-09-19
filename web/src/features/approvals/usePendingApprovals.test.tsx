import type { ReactNode } from "react";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as api from "@/api/approvals";

import {
  usePendingApprovals,
  usePendingCountInDocumentTitle,
} from "./usePendingApprovals";

function wrap() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("usePendingApprovals", () => {
  it("返回条数、条目与后端类型", async () => {
    vi.spyOn(api, "listPendingApprovals").mockResolvedValue({
      backend: "redis",
      items: [{ run_id: "r1", session_id: "s1", tool_name: "write" } as never],
    });
    const { result } = renderHook(() => usePendingApprovals(), {
      wrapper: wrap(),
    });
    await waitFor(() => expect(result.current.count).toBe(1));
    expect(result.current.backend).toBe("redis");
  });

  it("接口失败时条数为 0 且不抛（角标不该把页面搞崩）", async () => {
    vi.spyOn(api, "listPendingApprovals").mockRejectedValue(
      new Error("后端未启动"),
    );
    const { result } = renderHook(() => usePendingApprovals(), {
      wrapper: wrap(),
    });
    await waitFor(() => expect(result.current.count).toBe(0));
  });
});

describe("usePendingCountInDocumentTitle", () => {
  it("有条数时加前缀，归零时移除", async () => {
    document.title = "Agent 控制台";
    const { rerender } = renderHook(
      ({ n }: { n: number }) => usePendingCountInDocumentTitle(n),
      { initialProps: { n: 0 } },
    );
    expect(document.title).toBe("Agent 控制台");
    rerender({ n: 2 });
    expect(document.title).toBe("(2) Agent 控制台");
    rerender({ n: 0 });
    expect(document.title).toBe("Agent 控制台");
  });
});
