import { apiDelete, apiGet, apiPostForm, apiPostJson } from "@/lib/api";
import type { GraphCollection } from "./types";

export async function listGraphCollections(): Promise<GraphCollection[]> {
  return apiGet<GraphCollection[]>("/knowledgebase/collections");
}

export async function getGraphCollectionFiles(
  name: string,
): Promise<GraphCollection> {
  return apiGet<GraphCollection>(
    `/knowledgebase/collections/${encodeURIComponent(name)}/files`,
  );
}

/**
 * 单文件图谱上传（同步）。
 *
 * ⚠️ 路径是 `kownledgebase`（少一个 w）——这是后端的实际拼写，
 *   沿用其路由文件名，**不要"顺手改成正确拼写"**，改了就是 404。
 *   隔壁 `/documents/knowledgebase/upload-bulk` 才是正确拼写，但那是批量接口
 *   （202 后台任务，且后端没有进度查询接口），本期不使用。
 */
export async function uploadGraphDocument(
  file: File,
  collectionName: string,
  description: string,
): Promise<unknown> {
  const form = new FormData();
  form.append("file", file);
  form.append("collection_name", collectionName);
  form.append("description", description);
  return apiPostForm("/documents/kownledgebase/upload", form);
}

export async function deleteGraphCollectionFile(
  name: string,
  filename: string,
): Promise<unknown> {
  return apiDelete(
    `/knowledgebase/collections/${encodeURIComponent(name)}/files/${encodeURIComponent(filename)}`,
  );
}

export async function clearLegacyWorkspace(): Promise<unknown> {
  return apiPostJson("/knowledgebase/maintenance/clear-legacy-workspace", {});
}
