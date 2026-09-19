import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
// ⚠️ 从 vitest/config 导入，而不是 vite：这样 defineConfig 才知道 `test` 字段。
import { defineConfig } from "vitest/config";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // ⚠️ package.json 是 "type": "module"，本配置按 ESM 加载，
    //    `__dirname` 在 ESM 下不存在（会直接报错）。用 import.meta.url 推导。
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      // ⚠️ 开发期唯一让请求到达后端的东西。写错 → 页面正常但所有请求 404。
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
  },
});
