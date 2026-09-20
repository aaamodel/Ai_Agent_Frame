import { afterEach, describe, expect, it } from "vitest";

import { readMode, saveMode } from "./mode";

afterEach(() => localStorage.clear());

describe("mode", () => {
  it("默认是工作任务模式（本控制台的主用途）", () => {
    expect(readMode()).toBe("agent");
  });

  it("保存后能读回", () => {
    saveMode("chat");
    expect(readMode()).toBe("chat");
    saveMode("agent");
    expect(readMode()).toBe("agent");
  });

  it("存储值非法时退回默认，不让坏数据卡死界面", () => {
    localStorage.setItem("agent-console.mode", "bogus");
    expect(readMode()).toBe("agent");
  });
});
