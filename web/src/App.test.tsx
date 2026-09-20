import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

afterEach(() => vi.restoreAllMocks());

describe("App", () => {
  /**
   * 外壳级冒烟测试。
   *
   * ⚠️ 这里**不再**断言"Agent 控制台"那段文字 —— Task 1 时 App 是个占位 div，
   * 现在它已是路由表 + 外壳装配点，那段文字根本不存在了。
   * 改为断言左栏三个区的导航链接渲染出来，即"外壳装配成功"。
   *
   * ⚠️ 用 getByRole("link") 而不是 getByText(/对话/)：对话区的空状态里
   * 有"开始你的第一轮对话"，正则 /对话/ 会同时命中它和导航项，变成 multiple match。
   * 按 role+可访问名查询也更能表达"这是一条导航链接"这个真实意图。
   */
  it("装配出应用外壳（左栏三个区导航都渲染）", () => {
    render(<App />);
    expect(screen.getByRole("link", { name: "对话" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "文档" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "知识库" })).toBeInTheDocument();
  });
});
