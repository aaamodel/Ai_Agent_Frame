import { useQuery } from "@tanstack/react-query";

import { listGraphCollections } from "@/api/knowledgebase";

/**
 * 图谱集合列表 —— 左栏下半栏（知识库区）与图谱页共用。
 *
 * ⚠️ 固定 query key：左栏与页面共用缓存，切区不重新请求。
 */
export function useGraphCollectionList() {
  return useQuery({
    queryKey: ["graph-collections"],
    queryFn: listGraphCollections,
    retry: false,
  });
}
