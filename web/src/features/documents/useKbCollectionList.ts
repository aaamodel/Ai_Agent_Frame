import { useQuery } from "@tanstack/react-query";

import { listKbCollections } from "@/api/documents";

/**
 * RAG 集合列表 —— 左栏下半栏（文档区）与文档页共用。
 *
 * ⚠️ 固定 query key：左栏与页面共用同一份缓存，切区时不重新请求。
 *    （每个组件各自 useQuery 会造成请求风暴与角标闪烁，见 Review Focus #5）
 */
export function useKbCollectionList() {
  return useQuery({
    queryKey: ["kb-collections"],
    queryFn: listKbCollections,
    retry: false,
  });
}
