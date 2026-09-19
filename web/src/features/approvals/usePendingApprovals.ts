import { useEffect } from "react";

import { useQuery } from "@tanstack/react-query";

import { listPendingApprovals } from "@/api/approvals";
import type { PendingApproval } from "@/api/types";

const QUERY_KEY = ["pending-approvals"];

/**
 * 待审批角标的数据源。
 *
 * ⚠️ 这是"切到别的区也能看见挂起项"的**唯一**兜底——内联卡片只在对话区可见，
 * 而挂起是阻塞语义：没人处理，Agent 就一直在那儿等。
 *
 * ⚠️ 必须用**一个** query 实例（同一个 key）供左栏与标题共用。
 *    若各处各自 useQuery，切区时会反复重建、造成请求风暴与角标闪烁。
 */
export function usePendingApprovals(): {
  count: number;
  items: PendingApproval[];
  backend: string;
} {
  const { data } = useQuery({
    queryKey: QUERY_KEY,
    queryFn: listPendingApprovals,
    refetchInterval: 30_000,
    refetchOnWindowFocus: true,
    // 轮询失败（后端没起）不该让角标把页面搞崩，也不该打日志刷屏
    retry: false,
    throwOnError: false,
  });

  const items = data?.items ?? [];
  return { count: items.length, items, backend: data?.backend ?? "unknown" };
}

const BASE_TITLE = "Agent 控制台";

/** 把待审批条数写进浏览器标签页标题。 */
export function usePendingCountInDocumentTitle(count: number): void {
  useEffect(() => {
    document.title = count > 0 ? `(${count}) ${BASE_TITLE}` : BASE_TITLE;
  }, [count]);
}
