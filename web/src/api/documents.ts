import { apiDelete, apiGet, apiPostForm } from "@/lib/api";
import type { DocumentInfo, KbCollection, KbFile } from "./types";

/** 后端 `KbCollectionListResponse`（response_model）——**不是**裸数组。 */
interface KbCollectionListResponse {
  physical_collection: string;
  collections: KbCollection[];
}

/**
 * RAG 逻辑集合列表。
 *
 * ⚠️ 这个端点返回的是 ``{"physical_collection": ..., "collections": [...]}``，
 * **不是数组**。早期这里声明成 ``Promise<KbCollection[]>``，而 ``apiGet<T>``
 * 只是断言转型、不做运行时校验，TS 查不出来——页面一进文档区就崩在
 * ``(collections.data ?? []).map is not a function``。
 *
 * 这里既拆包装，也兼容"万一哪天后端改成裸数组"，避免再次崩在同一个地方。
 */
export async function listKbCollections(): Promise<KbCollection[]> {
  const raw = await apiGet<KbCollectionListResponse | KbCollection[]>(
    "/vector/collections",
  );
  return Array.isArray(raw) ? raw : (raw?.collections ?? []);
}

export async function getKbCollectionFiles(name: string): Promise<{
  name: string;
  description: string | null;
  document_count: number;
  files: KbFile[];
}> {
  return apiGet(`/vector/collections/${encodeURIComponent(name)}/files`);
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  return apiGet<DocumentInfo[]>("/documents");
}

export async function uploadDocument(
  file: File,
  collectionName: string,
  description: string,
): Promise<DocumentInfo> {
  const form = new FormData();
  form.append("file", file);
  form.append("collection_name", collectionName);
  form.append("description", description);
  return apiPostForm<DocumentInfo>("/documents/upload", form);
}

export async function deleteKbCollectionFile(
  name: string,
  filename: string,
): Promise<unknown> {
  // ⚠️ filename 必须编码：后端支持中文，但不编码会 404
  return apiDelete(
    `/vector/collections/${encodeURIComponent(name)}/files/${encodeURIComponent(filename)}`,
  );
}
