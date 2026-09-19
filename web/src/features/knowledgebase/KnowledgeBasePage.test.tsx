import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as api from "@/api/knowledgebase";

import { KnowledgeBasePage } from "./KnowledgeBasePage";

function renderAt(path: string) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/knowledgebase" element={<KnowledgeBasePage />} />
          <Route
            path="/knowledgebase/:collectionName"
            element={<KnowledgeBasePage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => vi.restoreAllMocks());

describe("KnowledgeBasePage", () => {
  it("列出图谱集合", async () => {
    vi.spyOn(api, "listGraphCollections").mockResolvedValue([
      {
        name: "Cross_Entity_Relation",
        legacy: false,
        description: "跨实体",
        document_count: 2,
        files: [],
      },
    ]);
    renderAt("/knowledgebase");
    await waitFor(() =>
      expect(screen.getByText("Cross_Entity_Relation")).toBeInTheDocument(),
    );
  });

  it("显示每个文件的处理状态", async () => {
    vi.spyOn(api, "getGraphCollectionFiles").mockResolvedValue({
      name: "Cross_Entity_Relation",
      legacy: false,
      description: "跨实体",
      document_count: 2,
      files: [
        { filename: "系统关系说明.pdf", status: "已处理", chunk_count: 86 },
        { filename: "组织架构.txt", status: "处理中", chunk_count: 0 },
      ],
    });
    renderAt("/knowledgebase/Cross_Entity_Relation");
    await waitFor(() => expect(screen.getByText("已处理")).toBeInTheDocument());
    expect(screen.getByText("处理中")).toBeInTheDocument();
  });

  /**
   * Important：图谱侧的字段名是 `chunks_count`（多一个 s），
   * 与 RAG 侧的 `chunk_count` 不同名。用错名字会让切片数恒为 0 / "—"。
   */
  it("用后端真实字段 chunks_count 显示切片数", async () => {
    vi.spyOn(api, "getGraphCollectionFiles").mockResolvedValue({
      name: "c",
      legacy: false,
      description: null,
      document_count: 1,
      files: [{ filename: "a.pdf", status: "已处理", chunks_count: 86 }],
    });
    renderAt("/knowledgebase/c");

    // 统计卡与表格行都会显示 86，故用 getAllByText
    await waitFor(() =>
      expect(screen.getAllByText("86").length).toBeGreaterThanOrEqual(2),
    );
    expect(screen.queryByText("—")).not.toBeInTheDocument();
  });

  it("如实说明无法查询进度（后端没有任务查询接口）", async () => {
    vi.spyOn(api, "getGraphCollectionFiles").mockResolvedValue({
      name: "c",
      legacy: false,
      description: null,
      document_count: 0,
      files: [],
    });
    renderAt("/knowledgebase/c");
    await waitFor(() =>
      expect(screen.getByText(/无法查询进度/)).toBeInTheDocument(),
    );
  });

  it("上传走单文件同步接口，不用批量接口", async () => {
    vi.spyOn(api, "getGraphCollectionFiles").mockResolvedValue({
      name: "c",
      legacy: false,
      description: null,
      document_count: 0,
      files: [],
    });
    const up = vi
      .spyOn(api, "uploadGraphDocument")
      .mockResolvedValue({} as never);
    renderAt("/knowledgebase/c");
    await waitFor(() => screen.getByLabelText(/选择文件/));

    const file = new File(["x"], "a.pdf", { type: "application/pdf" });
    await userEvent.upload(screen.getByLabelText(/选择文件/), file);
    await userEvent.click(screen.getByRole("button", { name: "上传" }));

    await waitFor(() => expect(up).toHaveBeenCalled());
  });
});
