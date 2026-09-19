import { apiDelete, apiGet, apiPostForm } from "@/lib/api";
import type { DocumentInfo, KbCollection, KbFile } from "./types";

export async function listKbCollections(): Promise<KbCollection[]> {
  return apiGet<KbCollection[]>("/vector/collections");
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
