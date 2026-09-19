import { afterEach, describe, expect, it, vi } from "vitest";

import { deleteKbCollectionFile, getKbCollectionFiles } from "./documents";

afterEach(() => vi.restoreAllMocks());

/** 拦下 fetch，返回成功响应，并暴露调用时用的 URL。 */
function spyFetch() {
  const spy = vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValue(new Response(JSON.stringify({}), { status: 200 }));
  return () => String(spy.mock.calls[0]?.[0] ?? "");
}

/**
 * Review Focus #3：文件名 / 集合名里的中文与空格必须编码。
 *
 * ⚠️ 编码动作发生在本模块（api 层），不在页面组件里 ——
 *    只断言组件传参是验不到编码的，必须看真实发出的 URL。
 */
describe("集合名与文件名的 URL 编码", () => {
  it("删除时对中文文件名做 URL 编码", async () => {
    const url = spyFetch();
    await deleteKbCollectionFile("sales_kb", "线索阶段流转规则.md");

    expect(url()).toBe(
      `/api/v1/vector/collections/sales_kb/files/${encodeURIComponent(
        "线索阶段流转规则.md",
      )}`,
    );
    // 未编码的原始中文不该出现在 URL 里
    expect(url()).not.toContain("线索阶段流转规则.md");
  });

  it("集合名含空格时同样编码", async () => {
    const url = spyFetch();
    await getKbCollectionFiles("客户 线索");

    expect(url()).toContain(encodeURIComponent("客户 线索"));
    expect(url()).not.toContain(" ");
  });

  it("删除时对集合名也编码", async () => {
    const url = spyFetch();
    await deleteKbCollectionFile("客户 线索", "a b.md");

    expect(url()).toBe(
      `/api/v1/vector/collections/${encodeURIComponent("客户 线索")}/files/${encodeURIComponent("a b.md")}`,
    );
  });
});
