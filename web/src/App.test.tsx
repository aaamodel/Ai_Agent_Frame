import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "./App";

describe("App", () => {
  it("渲染出控制台标题", () => {
    render(<App />);
    expect(screen.getByText("Agent 控制台")).toBeInTheDocument();
  });
});
