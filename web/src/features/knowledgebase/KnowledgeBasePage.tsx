import { useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import {
  clearLegacyWorkspace,
  deleteGraphCollectionFile,
  getGraphCollectionFiles,
  uploadGraphDocument,
} from "@/api/knowledgebase";

import { useGraphCollectionList } from "./useGraphCollectionList";

const STATUS_STYLE: Record<string, string> = {
  已处理: "text-ok-text",
  处理中: "text-warn-text",
  失败: "text-danger-text",
};

/**
 * 知识库区（图谱集合）—— 以「图谱建得怎么样」为中心。
 *
 * 与文档区**不共用组件**（用户决策）：图谱后续要单独加功能，
 * 与文档区绑定会导致改一处牵动另一处。
 */
export function KnowledgeBasePage() {
  const { collectionName } = useParams();
  const qc = useQueryClient();
  const collections = useGraphCollectionList();

  const detail = useQuery({
    queryKey: ["graph-files", collectionName],
    queryFn: () => getGraphCollectionFiles(collectionName!),
    enabled: Boolean(collectionName),
    retry: false,
  });

  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");

  const upload = useMutation({
    // ⚠️ 单文件同步接口（/documents/kownledgebase/upload，注意拼写）。
    // 批量接口返回 202，但后端没有任务查询端点，用了就只能"提交后刷新"。
    mutationFn: () =>
      uploadGraphDocument(pendingFile!, collectionName!, description),
    onSuccess: () => {
      setPendingFile(null);
      setDescription("");
      void qc.invalidateQueries({ queryKey: ["graph-files", collectionName] });
      void qc.invalidateQueries({ queryKey: ["graph-collections"] });
    },
  });

  const remove = useMutation({
    mutationFn: (filename: string) =>
      deleteGraphCollectionFile(collectionName!, filename),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["graph-files", collectionName] });
      void qc.invalidateQueries({ queryKey: ["graph-collections"] });
    },
  });

  const clearLegacy = useMutation({
    mutationFn: clearLegacyWorkspace,
    onSuccess: () =>
      void qc.invalidateQueries({ queryKey: ["graph-collections"] }),
  });

  if (!collectionName) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto p-6">
        <h1 className="mb-3 text-[18px] font-semibold text-fg">知识库</h1>
        <p className="mb-4 text-sm text-fg-muted">
          从左侧选择或点击下方任一图谱集合。图谱抽取在后台进行，
          构建完成后实体与关系才可用于检索。
        </p>
        <div className="overflow-hidden rounded-lg border border-line">
          <table className="w-full text-sm">
            <thead className="bg-surface-2 text-left text-fg-muted">
              <tr>
                <th className="px-3 py-2 font-medium">集合</th>
                <th className="px-3 py-2 font-medium">描述</th>
                <th className="px-3 py-2 font-medium">文件</th>
              </tr>
            </thead>
            <tbody>
              {(collections.data ?? []).map((c) => (
                <tr
                  key={c.name}
                  className="border-t border-line hover:bg-surface-2/60"
                >
                  <td className="px-3 py-2">
                    <Link
                      className="text-accent-text hover:underline"
                      to={`/knowledgebase/${encodeURIComponent(c.name)}`}
                    >
                      {c.name}
                    </Link>
                  </td>
                  <td className="px-3 py-2 text-fg-muted">
                    {c.description ?? "—"}
                  </td>
                  <td className="px-3 py-2 text-fg-muted">
                    {c.document_count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!collections.isLoading && (collections.data ?? []).length === 0 && (
          <div className="py-6 text-center text-sm text-fg-subtle">
            还没有图谱集合。上传文件时填写集合名即可创建。
          </div>
        )}
        <button
          className="mt-4 text-sm text-fg-subtle underline hover:text-fg-muted"
          onClick={() => {
            if (window.confirm("清空历史遗留 workspace？具名集合不受影响。")) {
              clearLegacy.mutate();
            }
          }}
        >
          清空历史遗留 workspace
        </button>
      </div>
    );
  }

  // ⚠️ 图谱侧的字段是 chunks_count（多一个 s），与 RAG 侧不同名
  const totalChunks = (detail.data?.files ?? []).reduce(
    (n, f) => n + (f.chunks_count ?? f.chunk_count ?? 0),
    0,
  );

  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-6">
      <h1 className="text-[18px] font-semibold text-fg">{collectionName}</h1>
      <p className="mb-3 text-sm text-fg-muted">
        {detail.data?.description ?? "（该集合暂无描述）"}
      </p>

      <div className="mb-5 flex gap-2 text-sm">
        <span className="rounded-lg bg-surface-3 px-3 py-1 text-fg-muted">
          文件 <strong className="text-fg">{detail.data?.document_count ?? 0}</strong>
        </span>
        <span className="rounded-lg bg-surface-3 px-3 py-1 text-fg-muted">
          切片 <strong className="text-fg">{totalChunks}</strong>
        </span>
      </div>

      <div className="mb-5 rounded-lg border border-line bg-surface-2 p-3">
        <label className="block text-sm font-medium text-fg" htmlFor="graph-file">
          选择文件
        </label>
        <input
          id="graph-file"
          type="file"
          accept=".pdf,.txt"
          className="mt-1 block text-sm text-fg-muted"
          onChange={(e) => setPendingFile(e.target.files?.[0] ?? null)}
        />
        <input
          className="mt-2 w-full rounded-lg border border-line bg-surface-3 px-2 py-1.5 text-sm text-fg placeholder:text-fg-subtle focus:border-accent-ring focus:outline-none"
          placeholder="集合描述（留空则不更新）"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
        <button
          className="mt-2 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
          disabled={!pendingFile || upload.isPending}
          onClick={() => upload.mutate()}
        >
          上传
        </button>
        {upload.isPending && (
          <span className="ml-2 text-sm text-fg-muted">
            正在解析并织入图谱…
          </span>
        )}
        {upload.isError && (
          <div className="mt-2 text-sm text-danger-text">
            上传失败：{(upload.error as Error).message}
          </div>
        )}
        <div className="mt-2 rounded-lg bg-surface-3 px-3 py-2 text-xs text-fg-muted">
          图谱抽取在后台进行。<strong className="text-fg">提交后无法查询进度</strong>
          （后端未提供任务查询接口），请稍后回来刷新本页。
        </div>
      </div>

      {detail.isError && (
        <div className="mb-3 rounded-lg border border-danger-border bg-danger-bg px-3 py-2 text-sm text-danger-text">
          加载失败：{(detail.error as Error).message}
          <button className="ml-2 underline" onClick={() => void detail.refetch()}>
            重试
          </button>
        </div>
      )}

      <div className="overflow-hidden rounded-lg border border-line">
        <table className="w-full text-sm">
          <thead className="bg-surface-2 text-left text-fg-muted">
            <tr>
              <th className="px-3 py-2 font-medium">文件</th>
              <th className="px-3 py-2 font-medium">处理状态</th>
              <th className="px-3 py-2 font-medium">切片</th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {(detail.data?.files ?? []).map((f) => (
              <tr
                key={f.filename}
                className="border-t border-line hover:bg-surface-2/60"
              >
                <td className="px-3 py-2">{f.filename}</td>
                <td
                  className={`px-3 py-2 ${
                    STATUS_STYLE[f.status ?? ""] ?? "text-fg-muted"
                  }`}
                >
                  {f.status ?? "—"}
                </td>
                <td className="px-3 py-2 text-fg-muted">
                  {f.chunks_count ?? f.chunk_count ?? "—"}
                </td>
                <td className="px-3 py-2 text-right">
                  <button
                    className="text-fg-subtle underline hover:text-danger-text"
                    onClick={() => {
                      if (window.confirm(`确认删除 ${f.filename}？`)) {
                        remove.mutate(f.filename);
                      }
                    }}
                  >
                    删除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!detail.isLoading && (detail.data?.files ?? []).length === 0 && (
        <div className="py-6 text-center text-sm text-fg-subtle">
          该集合还没有文件。
        </div>
      )}
    </div>
  );
}
