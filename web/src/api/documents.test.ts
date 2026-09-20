import { afterEach, describe, expect, it, vi } from "vitest";

import {
  deleteKbCollectionFile,
  getKbCollectionFiles,
  listKbCollections,
} from "./documents";

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

/**
 * 回归：``/vector/collections`` 返回的是 ``{"physical_collection", "collections"}``，
 * **不是裸数组**。
 *
 * ⚠️ 这条必须锁住：``apiGet<T>`` 只做断言转型，运行时不校验，所以把返回类型
 * 写成 ``Promise<KbCollection[]>`` 是**编译期查不出来**的谎——一进文档区就崩在
 * ``(collections.data ?? []).map is not a function``。
 */
describe("listKbCollections 的返回形态", () => {
  it("后端返回 {collections:[...]} 时拆出真正的数组", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          physical_collection: "knowledge_base_v3",
          collections: [
            { name: "sales_kb", description: "销售", document_count: 2 },
          ],
        }),
        { status: 200 },
      ),
    );

    const list = await listKbCollections();
    expect(Array.isArray(list)).toBe(true);
    expect(list).toHaveLength(1);
    expect(list[0]?.name).toBe("sales_kb");
  });

  it("后端若改成裸数组同样兼容，不再崩在同一个地方", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify([{ name: "x", description: null, document_count: 0 }]),
        { status: 200 },
      ),
    );
    expect(await listKbCollections()).toHaveLength(1);
  });

  it("collections 字段缺失时返回空数组，而不是把 undefined 交给页面", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ physical_collection: "p" }), {
        status: 200,
      }),
    );
    expect(await listKbCollections()).toEqual([]);
  });
});
