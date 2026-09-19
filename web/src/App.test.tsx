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
   * 改为断言左栏三个区的导航渲染出来，即"外壳装配成功"。
   */
  it("装配出应用外壳（左栏三个区导航都渲染）", () => {
    render(<App />);
    expect(screen.getByText(/对话/)).toBeInTheDocument();
    expect(screen.getByText(/文档/)).toBeInTheDocument();
    expect(screen.getByText(/知识库/)).toBeInTheDocument();
  });
});
