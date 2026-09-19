import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  createSession,
  deleteSession,
  deriveTitle,
  getSession,
  listSessions,
  newSessionId,
  saveSession,
} from "./sessions";

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("deriveTitle", () => {
  it("短文本原样", () => {
    expect(deriveTitle("行业优先级")).toBe("行业优先级");
  });

  it("超过 20 字符加省略号", () => {
    const t = deriveTitle("一二三四五六七八九十一二三四五六七八九十二三");
    expect(t).toBe("一二三四五六七八九十一二三四五六七八九十…");
    expect([...t].length).toBe(21); // 20 字符 + 省略号
  });

  it("空白输入退回默认标题", () => {
    expect(deriveTitle("   ")).toBe("新会话");
  });

  it("按码点计数，不把 emoji 切坏", () => {
    const t = deriveTitle("😀".repeat(25));
    expect([...t].length).toBe(21);
  });
});

describe("会话存储", () => {
  it("默认返回空列表", () => {
    expect(listSessions()).toEqual([]);
  });

  it("保存后能读回", () => {
    const s = { ...createSession(), title: "T" };
    saveSession(s);
    expect(getSession(s.id)?.title).toBe("T");
  });

  it("按 updatedAt 倒序排列", () => {
    // 直接写入存储，而不是用 saveSession 造数：
    // saveSession 会把 updatedAt 盖成"当前时间"（那是它的职责），
    // 连续两次保存会落在同一毫秒，排序结果不确定，测不出排序行为。
    localStorage.setItem(
      "agent-console.sessions",
      JSON.stringify([
        { id: "a", title: "旧", updatedAt: 1, messages: [] },
        { id: "b", title: "新", updatedAt: 2, messages: [] },
      ]),
    );
    expect(listSessions().map((s) => s.title)).toEqual(["新", "旧"]);
  });

  it("删除后读不到", () => {
    const s = createSession();
    saveSession(s);
    deleteSession(s.id);
    expect(getSession(s.id)).toBeUndefined();
  });

  it("localStorage 内容损坏时不崩，退回空列表", () => {
    localStorage.setItem("agent-console.sessions", "{不是JSON");
    expect(listSessions()).toEqual([]);
  });

  it("每次生成的会话 id 不同", () => {
    expect(newSessionId()).not.toBe(newSessionId());
  });
});
