import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as api from "@/api/documents";

import { DocumentsPage } from "./DocumentsPage";

function renderAt(path: string) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/documents/:collectionName" element={<DocumentsPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => vi.restoreAllMocks());

describe("DocumentsPage", () => {
  it("列出 RAG 集合", async () => {
    vi.spyOn(api, "listKbCollections").mockResolvedValue([
      { name: "sales_kb", description: "销售知识", document_count: 3 },
    ]);
    renderAt("/documents");
    await waitFor(() =>
      expect(screen.getByText("sales_kb")).toBeInTheDocument(),
    );
  });

  /**
   * 回归：``/vector/collections`` 的真实形态是 ``{physical_collection, collections}``。
   *
   * ⚠️ 这条**故意不 mock api 层**——现有用例都把 ``listKbCollections`` mock 成数组，
   *    与那个错误的类型声明是同一个谎，所以页面从来没在这里被真正验证过，
   *    用户一进 /documents 就 ``(collections.data ?? []).map is not a function``。
   *    这里只 mock 最底层的 fetch，让请求真的穿过 api 层的拆包装。
   */
  it("集合列表返回对象形态时页面照常渲染（真实链路回归）", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          physical_collection: "knowledge_base_v3",
          collections: [
            { name: "sales_kb", description: "销售知识", document_count: 3 },
          ],
        }),
        { status: 200 },
      ),
    );

    renderAt("/documents");
    await waitFor(() =>
      expect(screen.getByText("sales_kb")).toBeInTheDocument(),
    );
  });

  it("显示集合内文件与切片数", async () => {
    vi.spyOn(api, "getKbCollectionFiles").mockResolvedValue({
      name: "sales_kb",
      description: "销售知识",
      document_count: 2,
      files: [
        { filename: "年度目标.xlsx", chunk_count: 42 },
        { filename: "折扣权限与审批规则.md", chunk_count: 11 },
      ],
    });
    renderAt("/documents/sales_kb");
    await waitFor(() =>
      expect(screen.getByText("折扣权限与审批规则.md")).toBeInTheDocument(),
    );
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("删除时把原始文件名交给接口层（编码由 api 层负责）", async () => {
    // 编码本身由 api/documents.test.ts 验证；这里验组件不二次编码 / 不篡改文件名
    vi.spyOn(api, "getKbCollectionFiles").mockResolvedValue({
      name: "sales_kb",
      description: null,
      document_count: 1,
      files: [{ filename: "线索阶段流转规则.md", chunk_count: 1 }],
    });
    const del = vi
      .spyOn(api, "deleteKbCollectionFile")
      .mockResolvedValue({} as never);
    // 删除确认用 window.confirm（不引入 shadcn Dialog）
    vi.spyOn(window, "confirm").mockReturnValue(true);

    renderAt("/documents/sales_kb");
    await waitFor(() => screen.getByText("线索阶段流转规则.md"));
    await userEvent.click(screen.getAllByRole("button", { name: "删除" })[0]);

    expect(del).toHaveBeenCalledWith("sales_kb", "线索阶段流转规则.md");
  });

  it("取消确认时不发删除请求", async () => {
    vi.spyOn(api, "getKbCollectionFiles").mockResolvedValue({
      name: "sales_kb",
      description: null,
      document_count: 1,
      files: [{ filename: "a.md", chunk_count: 1 }],
    });
    const del = vi
      .spyOn(api, "deleteKbCollectionFile")
      .mockResolvedValue({} as never);
    vi.spyOn(window, "confirm").mockReturnValue(false);

    renderAt("/documents/sales_kb");
    await waitFor(() => screen.getByText("a.md"));
    await userEvent.click(screen.getAllByRole("button", { name: "删除" })[0]);

    expect(del).not.toHaveBeenCalled();
  });

  it("上传成功后刷新文件表", async () => {
    const files = vi.spyOn(api, "getKbCollectionFiles").mockResolvedValue({
      name: "sales_kb",
      description: null,
      document_count: 0,
      files: [],
    });
    vi.spyOn(api, "uploadDocument").mockResolvedValue({} as never);

    renderAt("/documents/sales_kb");
    await waitFor(() => expect(files).toHaveBeenCalled());

    const file = new File(["x"], "新规则.md", { type: "text/markdown" });
    await userEvent.upload(screen.getByLabelText(/选择文件/), file);
    await userEvent.click(screen.getByRole("button", { name: "上传" }));

    await waitFor(() => expect(api.uploadDocument).toHaveBeenCalled());
    // 上传后必须重新拉取，否则用户看不到刚传的文件
    await waitFor(() => expect(files.mock.calls.length).toBeGreaterThan(1));
  });
});
