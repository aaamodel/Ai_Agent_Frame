import { useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import {
  deleteKbCollectionFile,
  getKbCollectionFiles,
  uploadDocument,
} from "@/api/documents";

import { useKbCollectionList } from "./useKbCollectionList";

/**
 * 文档区（RAG 集合管理）—— 以「文档」为中心。
 *
 * 与知识库区**不共用组件**（用户决策）：两块后续会各自演进。
 */
export function DocumentsPage() {
  const { collectionName } = useParams();
  const qc = useQueryClient();
  // 左栏与页面共用同一份缓存（固定 query key）
  const collections = useKbCollectionList();

  const files = useQuery({
    queryKey: ["kb-files", collectionName],
    queryFn: () => getKbCollectionFiles(collectionName!),
    enabled: Boolean(collectionName),
    retry: false,
  });

  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");

  const upload = useMutation({
    mutationFn: () => uploadDocument(pendingFile!, collectionName!, description),
    onSuccess: () => {
      setPendingFile(null);
      setDescription("");
      // 上传后必须失效缓存，否则文件表看不到刚传的东西
      void qc.invalidateQueries({ queryKey: ["kb-files", collectionName] });
      void qc.invalidateQueries({ queryKey: ["kb-collections"] });
      void qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  const remove = useMutation({
    mutationFn: (filename: string) =>
      deleteKbCollectionFile(collectionName!, filename),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["kb-files", collectionName] });
      void qc.invalidateQueries({ queryKey: ["kb-collections"] });
    },
  });

  if (!collectionName) {
    return (
      <div className="p-6">
        <h1 className="mb-4 text-lg font-semibold">文档</h1>
        <p className="mb-4 text-sm text-neutral-500">
          从左侧选择或点击下方任一集合，查看其文件并上传。
        </p>
        {collections.isError && (
          <div className="mb-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            集合列表加载失败：{(collections.error as Error).message}
            <button
              className="ml-2 underline"
              onClick={() => void collections.refetch()}
            >
              重试
            </button>
          </div>
        )}
        <table className="w-full text-sm">
          <thead className="bg-neutral-50 text-left text-neutral-500">
            <tr>
              <th className="px-3 py-2">集合</th>
              <th className="px-3 py-2">描述</th>
              <th className="px-3 py-2">文件</th>
            </tr>
          </thead>
          <tbody>
            {(collections.data ?? []).map((c) => (
              <tr key={c.name} className="border-t border-neutral-100">
                <td className="px-3 py-2">
                  <Link
                    className="text-blue-600 hover:underline"
                    to={`/documents/${encodeURIComponent(c.name)}`}
                  >
                    {c.name}
                  </Link>
                </td>
                <td className="px-3 py-2 text-neutral-500">
                  {c.description ?? "—"}
                </td>
                <td className="px-3 py-2 text-neutral-500">
                  {c.document_count}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!collections.isLoading && (collections.data ?? []).length === 0 && (
          <div className="py-6 text-center text-sm text-neutral-400">
            还没有集合。上传文档时填写集合名即可创建。
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold">{collectionName}</h1>
      <p className="mb-4 text-sm text-neutral-500">
        {files.data?.description ?? "（该集合暂无描述）"}
      </p>

      <div className="mb-5 rounded border border-neutral-200 p-3">
        <label className="block text-sm font-medium" htmlFor="kb-file">
          选择文件
        </label>
        <input
          id="kb-file"
          type="file"
          className="mt-1 block text-sm"
          onChange={(e) => setPendingFile(e.target.files?.[0] ?? null)}
        />
        <input
          className="mt-2 w-full rounded border border-neutral-300 px-2 py-1 text-sm"
          placeholder="集合描述（留空则不更新；这个描述会影响后续意图路由）"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
        <button
          className="mt-2 rounded bg-blue-600 px-3 py-1 text-sm text-white disabled:opacity-40"
          disabled={!pendingFile || upload.isPending}
          onClick={() => upload.mutate()}
        >
          上传
        </button>
        {upload.isError && (
          <div className="mt-2 text-sm text-red-600">
            上传失败：{(upload.error as Error).message}
          </div>
        )}
      </div>

      {files.isError && (
        <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          加载失败：{(files.error as Error).message}
          <button className="ml-2 underline" onClick={() => void files.refetch()}>
            重试
          </button>
        </div>
      )}

      <table className="w-full text-sm">
        <thead className="bg-neutral-50 text-left text-neutral-500">
          <tr>
            <th className="px-3 py-2">文件</th>
            <th className="px-3 py-2">切片</th>
            <th className="px-3 py-2" />
          </tr>
        </thead>
        <tbody>
          {(files.data?.files ?? []).map((f) => (
            <tr key={f.filename} className="border-t border-neutral-100">
              <td className="px-3 py-2">{f.filename}</td>
              <td className="px-3 py-2 text-neutral-500">
                {f.chunk_count ?? f.chunks ?? "—"}
              </td>
              <td className="px-3 py-2 text-right">
                <button
                  className="text-neutral-500 underline"
                  onClick={() => {
                    // 二次确认：删除是破坏性操作且不可撤销
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
      {!files.isLoading && (files.data?.files ?? []).length === 0 && (
        <div className="py-6 text-center text-sm text-neutral-400">
          该集合还没有文件。
        </div>
      )}
    </div>
  );
}
