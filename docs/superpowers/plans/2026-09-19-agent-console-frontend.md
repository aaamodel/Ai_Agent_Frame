# Agent 控制台前端 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `Enterprise_aiagent` 后端建一个 React 前端控制台，覆盖对话（含危险工具审批与降级提示）、RAG 文档管理、知识图谱管理三个区。

**Architecture:** 前端独立在顶层 `web/`，用 Vite + React 19 + Tailwind v4 + shadcn/ui + TanStack Query + react-router。后端仅在 `app/main.py` 增加静态挂载（`web/dist` + SPA 回退），实现前后端同源、零 CORS。对话走 `fetch` + `ReadableStream` 解析自定义 SSE（**不能用 `EventSource`，因为接口是 POST**），审批是**循环**而非一次性交互。

**Tech Stack:** Vite 8 · React 19 · TypeScript · Tailwind CSS 4 · shadcn/ui (CLI 4) · TanStack Query 5 · react-router 7 · react-markdown · Vitest 5 · Testing Library

**Spec:** `docs/superpowers/specs/2026-09-19-agent-console-frontend-design.md`

## Global Constraints

以下约束适用于**每一个**任务，各任务的 requirements 隐含包含本节。

- **API 前缀固定 `/api/v1`**（`app/config.py:178`）。所有请求路径以此开头。
- **路径拼写陷阱：两个上传接口拼法不同，且都真实存在，禁止"统一"。**
  - 单文件同步：`POST /api/v1/documents/kownledgebase/upload`（沿用文件名 `kownledgebase.py` 的拼写）
  - 批量 202：`POST /api/v1/documents/knowledgebase/upload-bulk`（正确拼写）
- **SSE 必须用 `fetch` + `ReadableStream`。** 接口是 POST，`EventSource` 是 GET-only，用不了。
- **Tailwind 是 v4：不存在 `tailwind.config.js`。** 配置写在入口 CSS 的 `@theme { … }` 里。按 v3 写法建配置文件会**静默失效**（不报错，样式不生效）。
- **react-router 只用 `react-router-dom` 一个入口，不与 `react-router` 混用**（混用会产生两份 router 上下文，表现为导航不生效）。
- **不引入：** Redux / Zustand / MUI / AntD / Assistant UI / 任何其他 UI 框架或 AI 聊天组件库。
- **`strategy` 固定用后端默认 `auto`，界面不暴露。**
- **待审批轮询间隔固定 30 秒**；窗口重新获得焦点时立即刷一次。
- **会话标题取首条用户消息前 20 个字符**，超出加 `…`；无用户消息时为"新会话"。
- **所有含中文 / 空格 / 路径分隔符的文件名必须 `encodeURIComponent`**（后端 `{filename:path}` 支持中文，但需 URL 编码）。
- **前端目录为 `web/`，与 `app/` 平级。** 不放进 `app/`。
- **后端只允许改 `app/main.py`（静态挂载，约 5 行）。** 不改任何 router。
- **后端既有 376 项测试必须保持全绿**（另有 1 项既有失败与本变更无关）。
- 每个任务结束都要 commit。

## Review Focus

以下是规格隐含、但没有任务测试会主动覆盖的失效模式，按"最可能咬人"排序。
**每条都在下方指定任务里加了对应测试**，不要跳过。

1. **SSE 事件被 TCP 分片从中间切开**（`data: ` 与 `\n\n` 落在不同 chunk，或一个 chunk 里塞了多条事件）。
   用户看到的是**回答被截断**或整条流报 JSON 解析错。→ 见 Task 2 的解析器测试。
2. **后端重启后用户才点"批准"** → `run_not_found`。界面若静默失败或只显示通用错误，
   用户会以为"点了没反应"，反复点。→ 见 Task 6 的失效分支测试。
3. **中文 / 带空格 / 带 `/` 的文件名删除失败**（忘记 URL 编码）→ 404 且提示含糊。→ 见 Task 10 的编码测试。
4. **dev proxy 配错**（`vite.config.ts` 里路径写错）→ 页面正常渲染但**所有**请求 404，
   新手容易误判成"后端没写对"。→ 见 Task 1 的验收步骤。
5. **左栏轮询在切区时被反复重建** → 请求风暴、角标闪烁。→ 见 Task 9 的单一查询实例测试。

---

### Task 1: 前端工程骨架

**Files:**
- Create: `web/`（由 Vite 脚手架生成，含 `package.json` / `vite.config.ts` / `tsconfig.json` / `index.html`）
- Create: `web/src/main.tsx` / `web/src/App.tsx`
- Create: `web/src/index.css`
- Create: `web/vitest.setup.ts`
- Modify: `web/vite.config.ts`（加 proxy 与 vitest 配置）
- Create: `web/src/App.test.tsx`
- Modify: `.gitignore`（忽略 `web/node_modules/` 与 `web/dist/`）

**Interfaces:**
- Consumes: 无（第一个任务）
- Produces: 可运行的 `npm run dev` / `npm run build` / `npm test`；`@/` 路径别名指向 `web/src`

- [ ] **Step 1: 生成 Vite 工程**

```bash
cd "d:/pycharm/PyCharm 2026.1.1/PythonProject_deepagents/Enterprise_aiagent"
npm create vite@latest web -- --template react-ts
cd web && npm install
```

- [ ] **Step 2: 装依赖**

```bash
npm install react-router-dom @tanstack/react-query react-markdown remark-gfm
npm install -D tailwindcss @tailwindcss/vite vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom
```

⚠️ 装的是 `@tailwindcss/vite`（v4 的 Vite 插件），**不是** `tailwindcss` 的 PostCSS 插件写法。

- [ ] **Step 3: shadcn 初始化**

```bash
npx shadcn@latest init
```

选择：`New York` 风格、`zinc` 基色、CSS 变量开启。

⚠️ shadcn 会改写 `web/src/index.css`，**在 Step 4 之前完成本步**，否则会覆盖你手写的 CSS。

- [ ] **Step 4: 写入口 CSS（Tailwind v4 写法）**

`web/src/index.css`（替换 shadcn 生成内容，保留其 `:root` 变量块）：

```css
@import "tailwindcss";

/* ⚠️ Tailwind v4：配置写在这里，不存在 tailwind.config.js。
   若你发现样式完全没生效，第一个要检查的就是这一行 @import。 */
@theme {
  --color-agent-primary: oklch(0.55 0.22 265);
  --color-warn-bg: oklch(0.98 0.03 85);
  --color-warn-border: oklch(0.78 0.15 65);
  --color-warn-text: oklch(0.45 0.13 55);
  --color-ok-bg: oklch(0.97 0.03 155);
  --color-ok-border: oklch(0.75 0.16 155);
  --color-ok-text: oklch(0.42 0.12 155);
}

html, body, #root { height: 100%; }
```

- [ ] **Step 5: 写 `vite.config.ts`（代理 + 别名 + vitest）**

```ts
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    proxy: {
      // ⚠️ 这是开发期唯一让请求到达后端的东西。写错 → 页面正常但所有请求 404。
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
  },
});
```

- [ ] **Step 6: 写 `vitest.setup.ts`**

```ts
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 7: 写占位 `App.tsx` 与 `main.tsx`**

`web/src/App.tsx`：

```tsx
export default function App() {
  return <div className="p-8 text-2xl font-semibold">Agent 控制台</div>;
}
```

`web/src/main.tsx`：

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

- [ ] **Step 8: 写第一个测试**

`web/src/App.test.tsx`：

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("渲染出控制台标题", () => {
    render(<App />);
    expect(screen.getByText("Agent 控制台")).toBeInTheDocument();
  });
});
```

- [ ] **Step 9: 跑测试**

Run: `cd web && npm test -- --run`
Expected: PASS（1 passed）

- [ ] **Step 10: 补 `.gitignore`**

在仓库根 `.gitignore` 追加：

```
# 前端
web/node_modules/
web/dist/
```

- [ ] **Step 11: 人工验收 dev proxy**

```bash
# 终端 1
uvicorn app.main:app --reload
# 终端 2
cd web && npm run dev
```

浏览器打开 Vite 提示的地址（默认 `http://localhost:5173`），控制台执行：

```js
await fetch("/api/v1/health").then(r => r.status)
```

Expected: 返回 `200`（或后端实际的 health 状态码），**不是** 404。
若返回 404 → `vite.config.ts` 的 `server.proxy` 没生效，检查 `"/api"` 键名与后端端口。

- [ ] **Step 12: Commit**

```bash
git add web .gitignore
git commit -m "feat(web): 前端工程骨架（Vite+React+Tailwind v4+shadcn+Vitest）"
```

---

### Task 2: SSE 解析器（`lib/sse.ts`）

**Files:**
- Create: `web/src/lib/sse.ts`
- Test: `web/src/lib/sse.test.ts`

**Interfaces:**
- Consumes: 无
- Produces:
  - `parseSSEStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<Record<string, unknown>>`
    —— 逐条产出解析后的 JSON 事件对象；无法解析的事件被**跳过**（不抛错，不中断流）

- [ ] **Step 1: 写失败测试（覆盖 Review Focus #1）**

`web/src/lib/sse.test.ts`：

```ts
import { describe, expect, it } from "vitest";
import { parseSSEStream } from "./sse";

/** 把若干字符串分片喂给解析器，收集产出的事件。 */
async function collect(chunks: string[]) {
  const enc = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  const out: Record<string, unknown>[] = [];
  for await (const ev of parseSSEStream(stream)) out.push(ev);
  return out;
}

describe("parseSSEStream", () => {
  it("解析单条完整事件", async () => {
    const events = await collect(['data: {"content":"你好"}\n\n']);
    expect(events).toEqual([{ content: "你好" }]);
  });

  it("把跨 chunk 被切开的同一条事件拼回来", async () => {
    // ⚠️ Review Focus #1：TCP 分片会从任意位置切开，包括 JSON 中间
    const events = await collect([
      'data: {"cont',
      'ent":"被切开',
      '"}\n',
      "\n",
    ]);
    expect(events).toEqual([{ content: "被切开" }]);
  });

  it("一个 chunk 里塞多条事件时全部产出", async () => {
    const events = await collect([
      'data: {"content":"a"}\n\ndata: {"content":"b"}\n\ndata: {"done":true}\n\n',
    ]);
    expect(events).toEqual([{ content: "a" }, { content: "b" }, { done: true }]);
  });

  it("跳过解析不了的事件，但继续处理后续事件", async () => {
    // 不允许因一条坏事件中断整条流——否则用户看到回答莫名截断
    const events = await collect([
      "data: 这不是JSON\n\n",
      'data: {"content":"后续仍在"}\n\n',
    ]);
    expect(events).toEqual([{ content: "后续仍在" }]);
  });

  it("流结束时残留的半条事件被丢弃而不抛错", async () => {
    const events = await collect(['data: {"content":"未闭合']);
    expect(events).toEqual([]);
  });

  it("空 data 行不产出事件", async () => {
    const events = await collect(["\n\n", 'data: {"content":"x"}\n\n']);
    expect(events).toEqual([{ content: "x" }]);
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/lib/sse.test.ts`
Expected: FAIL，报错含 `Failed to resolve import "./sse"` 或 `parseSSEStream is not a function`

- [ ] **Step 3: 实现**

`web/src/lib/sse.ts`：

```ts
/**
 * 解析本项目后端自定义的 SSE 流。
 *
 * 后端格式（app/api/routes/chat.py 的 `_sse_payload`）：
 *     data: {"content":"…"}\n\n
 *
 * ⚠️ 不能用 `EventSource`：它是 GET-only，而 /chat/with_agent 是 POST。
 * 因此这里手写解析，必须自己处理**事件跨 chunk 被切开**的情况。
 */
export async function* parseSSEStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<Record<string, unknown>> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      // stream:true 保证多字节 UTF-8 字符被切开时也能正确拼回
      buffer += decoder.decode(value, { stream: true });

      // 事件以空行分隔。注意用 \n\n，且要处理 \r\n\n 这类变体。
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const parsed = parseOneEvent(rawEvent);
        if (parsed) yield parsed;
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
  // 结束时 buffer 里残留的半条事件**有意丢弃**：
  // 它意味着连接在事件中途断了，半条 JSON 无法可靠复原。
}

function parseOneEvent(rawEvent: string): Record<string, unknown> | null {
  const dataLines = rawEvent
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());

  if (dataLines.length === 0) return null;
  const payload = dataLines.join("\n").trim();
  if (!payload) return null;

  try {
    const parsed: unknown = JSON.parse(payload);
    // 只接受对象——数组/字符串/数字都不是本后端的合法事件
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    // 坏事件跳过而非抛出：一条解析不了的事件不该让整条回答消失
    return null;
  }
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd web && npm test -- --run src/lib/sse.test.ts`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/sse.ts web/src/lib/sse.test.ts
git commit -m "feat(web): 自定义 SSE 解析器（含跨 chunk 分片与坏事件容错）"
```

---

### Task 3: HTTP 封装与接口类型（`lib/api.ts`）

**Files:**
- Create: `web/src/lib/api.ts`
- Test: `web/src/lib/api.test.ts`

**Interfaces:**
- Consumes: 无
- Produces:
  - `class ApiError extends Error { status: number; detail: string }`
  - `apiGet<T>(path: string): Promise<T>`
  - `apiDelete<T>(path: string): Promise<T>`
  - `apiPostForm<T>(path: string, form: FormData): Promise<T>`
  - `apiPostJson<T>(path: string, body: unknown): Promise<T>`
  - `apiPostStream(path: string, body: unknown, signal?: AbortSignal): Promise<ReadableStream<Uint8Array>>`
  - `API_BASE = "/api/v1"`

- [ ] **Step 1: 写失败测试**

`web/src/lib/api.test.ts`：

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiGet, apiPostJson, API_BASE } from "./api";

afterEach(() => vi.restoreAllMocks());

describe("api 封装", () => {
  it("API_BASE 以 /api/v1 开头", () => {
    expect(API_BASE).toBe("/api/v1");
  });

  it("GET 拼出正确 URL 并解析 JSON", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: 1 }), { status: 200 }),
    );
    await expect(apiGet("/documents")).resolves.toEqual({ ok: 1 });
    expect(spy).toHaveBeenCalledWith("/api/v1/documents", expect.anything());
  });

  it("非 2xx 抛 ApiError 并带上 detail", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "集合不存在或为空" }), { status: 404 }),
    );
    const err = await apiGet("/vector/collections/nope/files").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(404);
    expect(err.detail).toBe("集合不存在或为空");
  });

  it("响应不是 JSON 时不崩，detail 退回状态文本", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("<html>500</html>", { status: 500 }),
    );
    const err = await apiGet("/documents").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(500);
    expect(err.detail.length).toBeGreaterThan(0);
  });

  it("POST JSON 带上 Content-Type 与序列化后的 body", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({}), { status: 200 }),
    );
    await apiPostJson("/agent/runs/x/approval", { approved: true, comment: "ok" });
    const [, init] = spy.mock.calls[0];
    expect(init?.method).toBe("POST");
    expect((init?.headers as Record<string, string>)["Content-Type"]).toContain(
      "application/json",
    );
    expect(init?.body).toBe('{"approved":true,"comment":"ok"}');
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/lib/api.test.ts`
Expected: FAIL，含 `Failed to resolve import "./api"`

- [ ] **Step 3: 实现**

`web/src/lib/api.ts`：

```ts
/** 后端 API 前缀（app/config.py 的 api_prefix）。 */
export const API_BASE = "/api/v1";

/** 带状态码与后端 detail 的请求错误。 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/** 从响应体里尽力取出可读的错误说明。 */
async function readDetail(res: Response): Promise<string> {
  const fallback = res.statusText || `HTTP ${res.status}`;
  try {
    const text = await res.text();
    if (!text) return fallback;
    try {
      const parsed: unknown = JSON.parse(text);
      if (parsed && typeof parsed === "object" && "detail" in parsed) {
        return String((parsed as { detail: unknown }).detail ?? fallback);
      }
    } catch {
      // 后端偶尔返回 HTML 错误页（如 502），退回状态文本
    }
    return fallback;
  } catch {
    return fallback;
  }
}

async function ensureOk(res: Response): Promise<Response> {
  if (!res.ok) throw new ApiError(res.status, await readDetail(res));
  return res;
}

async function asJson<T>(res: Response): Promise<T> {
  await ensureOk(res);
  return (await res.json()) as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  return asJson<T>(await fetch(`${API_BASE}${path}`, { method: "GET" }));
}

export async function apiDelete<T>(path: string): Promise<T> {
  return asJson<T>(await fetch(`${API_BASE}${path}`, { method: "DELETE" }));
}

export async function apiPostJson<T>(path: string, body: unknown): Promise<T> {
  return asJson<T>(
    await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  // ⚠️ 不要手工设 Content-Type：multipart 的 boundary 必须由浏览器生成
  return asJson<T>(
    await fetch(`${API_BASE}${path}`, { method: "POST", body: form }),
  );
}

/**
 * 发起一个返回 SSE 流的 POST，返回**未解析**的字节流。
 *
 * 解析交给 `lib/sse.ts`——这里只负责拿到流并把非 2xx 转成 ApiError。
 * 注意不能在这里 `await res.json()`，否则会把流读完，后面就再也读不到了。
 */
export async function apiPostStream(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  await ensureOk(res);
  if (!res.body) throw new ApiError(res.status, "响应没有可读的流");
  return res.body;
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd web && npm test -- --run src/lib/api.test.ts`
Expected: PASS（5 passed）

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/api.ts web/src/lib/api.test.ts
git commit -m "feat(web): HTTP 封装（错误规整 + POST 流）"
```

---

### Task 4: 接口层与路径常量

**Files:**
- Create: `web/src/api/types.ts`
- Create: `web/src/api/documents.ts`
- Create: `web/src/api/knowledgebase.ts`
- Create: `web/src/api/approvals.ts`
- Create: `web/src/api/chat.ts`
- Test: `web/src/api/paths.test.ts`

**Interfaces:**
- Consumes: `lib/api.ts` 的 `apiGet` / `apiDelete` / `apiPostForm` / `apiPostJson` / `apiPostStream`
- Produces:
  - `api/types.ts`：`KbCollection` / `KbFile` / `DocumentInfo` / `GraphCollection` / `GraphFile` / `PendingApproval` / `RunStatus` / `ApprovalDecision`
  - `documents.ts`：`listKbCollections()` / `getKbCollectionFiles(name)` / `listDocuments()` / `uploadDocument(file, collectionName, description)` / `deleteKbCollectionFile(name, filename)`
  - `knowledgebase.ts`：`listGraphCollections()` / `getGraphCollectionFiles(name)` / `uploadGraphDocument(file, collectionName, description)` / `deleteGraphCollectionFile(name, filename)` / `clearLegacyWorkspace()`
  - `approvals.ts`：`listPendingApprovals()` / `getRunStatus(runId)` / `apiResumeRun(runId, decision)`
  - `chat.ts`：`streamAgentChat(query, sessionId): Promise<ReadableStream<Uint8Array>>`

- [ ] **Step 1: 写失败测试（钉死路径，覆盖拼写陷阱）**

`web/src/api/paths.test.ts`：

```ts
import { describe, expect, it } from "vitest";
import * as kb from "./knowledgebase";
import * as docs from "./documents";
import * as approvals from "./approvals";

describe("接口路径常量", () => {
  it("单文件图谱上传沿用 kownledgebase 拼写（这是后端实际路径，不要改）", async () => {
    // ⚠️ Review Focus：后端两个上传接口拼法不同且都真实存在，统一即 404
    const src = await import("./knowledgebase");
    const text = JSON.stringify(src) + String(src.uploadGraphDocument);
    expect(text).toContain("kownledgebase");
  });

  it("图谱上传不走批量接口（本期已决定不做批量上传）", () => {
    const text = String(kb.uploadGraphDocument);
    expect(text).not.toContain("upload-bulk");
  });

  it("RAG 上传走 /documents/upload", () => {
    expect(String(docs.uploadDocument)).toContain("/documents/upload");
  });

  it("审批恢复走 /agent/runs/{id}/approval", () => {
    expect(String(approvals.apiResumeRun)).toContain("/approval");
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/api/paths.test.ts`
Expected: FAIL，`Failed to resolve import "./knowledgebase"`

- [ ] **Step 3: 写 `api/types.ts`**

```ts
/** 后端返回结构的前端映射。字段名与后端保持一字不差。 */

export interface KbCollection {
  name: string;
  description: string | null;
  document_count: number;
}

export interface KbFile {
  filename: string;
  chunk_count?: number;
  /** 后端在不同接口里用的字段名不完全一致，两个都留着 */
  chunks?: number;
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  collection_name: string;
  description?: string | null;
}

export interface GraphFile {
  filename: string;
  /** LightRAG 处理状态：已处理 / 处理中 / 失败 */
  status?: string;
  chunk_count?: number;
}

export interface GraphCollection {
  name: string;
  legacy: boolean;
  description: string | null;
  document_count: number;
  files: GraphFile[];
}

/** GET /agent/approvals/pending 的单条记录。 */
export interface PendingApproval {
  run_id: string;
  session_id: string;
  trace_id?: string;
  query?: string;
  mode?: string;
  tool_name?: string;
  args_preview?: string;
  subtask_id?: string;
  paused_at?: string;
}

export interface PendingApprovalsResponse {
  /** "redis" | "memory" —— memory 表示后端重启后挂起项失效 */
  backend: string;
  items: PendingApproval[];
}

/** GET /agent/runs/{run_id} 的快照。 */
export interface RunStatus {
  exists: boolean;
  paused?: boolean;
  next?: unknown;
  interrupts?: unknown;
  mode_used?: string;
  success?: boolean;
}

export interface ApprovalDecision {
  approved: boolean;
  comment: string;
}
```

- [ ] **Step 4: 写 `api/documents.ts`**

```ts
import { apiDelete, apiGet, apiPostForm } from "@/lib/api";
import type { DocumentInfo, KbCollection, KbFile } from "./types";

export async function listKbCollections(): Promise<KbCollection[]> {
  return apiGet<KbCollection[]>("/vector/collections");
}

export async function getKbCollectionFiles(name: string): Promise<{
  name: string;
  description: string | null;
  document_count: number;
  files: KbFile[];
}> {
  return apiGet(`/vector/collections/${encodeURIComponent(name)}/files`);
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  return apiGet<DocumentInfo[]>("/documents");
}

export async function uploadDocument(
  file: File,
  collectionName: string,
  description: string,
): Promise<DocumentInfo> {
  const form = new FormData();
  form.append("file", file);
  form.append("collection_name", collectionName);
  form.append("description", description);
  return apiPostForm<DocumentInfo>("/documents/upload", form);
}

export async function deleteKbCollectionFile(
  name: string,
  filename: string,
): Promise<unknown> {
  // ⚠️ filename 必须编码：后端支持中文，但不编码会 404
  return apiDelete(
    `/vector/collections/${encodeURIComponent(name)}/files/${encodeURIComponent(filename)}`,
  );
}
```

- [ ] **Step 5: 写 `api/knowledgebase.ts`**

```ts
import { apiDelete, apiGet, apiPostForm, apiPostJson } from "@/lib/api";
import type { GraphCollection } from "./types";

export async function listGraphCollections(): Promise<GraphCollection[]> {
  return apiGet<GraphCollection[]>("/knowledgebase/collections");
}

export async function getGraphCollectionFiles(
  name: string,
): Promise<GraphCollection> {
  return apiGet<GraphCollection>(
    `/knowledgebase/collections/${encodeURIComponent(name)}/files`,
  );
}

/**
 * 单文件图谱上传（同步）。
 *
 * ⚠️ 路径是 `kownledgebase`（少一个 w）——这是后端的实际拼写，
 *   沿用其路由文件名，**不要"顺手改成正确拼写"**，改了就是 404。
 *   隔壁 `/documents/knowledgebase/upload-bulk` 才是正确拼写，但那是批量接口
 *   （202 后台任务，且后端没有进度查询接口），本期不使用。
 */
export async function uploadGraphDocument(
  file: File,
  collectionName: string,
  description: string,
): Promise<unknown> {
  const form = new FormData();
  form.append("file", file);
  form.append("collection_name", collectionName);
  form.append("description", description);
  return apiPostForm("/documents/kownledgebase/upload", form);
}

export async function deleteGraphCollectionFile(
  name: string,
  filename: string,
): Promise<unknown> {
  return apiDelete(
    `/knowledgebase/collections/${encodeURIComponent(name)}/files/${encodeURIComponent(filename)}`,
  );
}

export async function clearLegacyWorkspace(): Promise<unknown> {
  return apiPostJson("/knowledgebase/maintenance/clear-legacy-workspace", {});
}
```

- [ ] **Step 6: 写 `api/approvals.ts`**

```ts
import { apiGet, apiPostStream } from "@/lib/api";
import type {
  ApprovalDecision,
  PendingApprovalsResponse,
  RunStatus,
} from "./types";

export async function listPendingApprovals(): Promise<PendingApprovalsResponse> {
  return apiGet<PendingApprovalsResponse>("/agent/approvals/pending");
}

export async function getRunStatus(runId: string): Promise<RunStatus> {
  return apiGet<RunStatus>(`/agent/runs/${encodeURIComponent(runId)}`);
}

/** 提交审批决定并拿到续流（SSE）。 */
export async function apiResumeRun(
  runId: string,
  decision: ApprovalDecision,
  signal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  return apiPostStream(
    `/agent/runs/${encodeURIComponent(runId)}/approval`,
    decision,
    signal,
  );
}
```

- [ ] **Step 7: 写 `api/chat.ts`**

```ts
import { apiPostStream } from "@/lib/api";

/**
 * 发起 Agent 对话，拿到 SSE 字节流。
 *
 * `strategy` 固定 `auto`（本设计不向用户暴露该参数）。
 */
export async function streamAgentChat(
  query: string,
  sessionId: string,
  signal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  return apiPostStream(
    "/chat/with_agent",
    { query, session_id: sessionId, strategy: "auto" },
    signal,
  );
}
```

- [ ] **Step 8: 跑测试确认通过**

Run: `cd web && npm test -- --run src/api/paths.test.ts`
Expected: PASS（4 passed）

- [ ] **Step 9: Commit**

```bash
git add web/src/api
git commit -m "feat(web): 接口层与类型（钉死 kownledgebase 路径拼写）"
```

---

### Task 5: SSE 事件类型与流处理助手

**Files:**
- Create: `web/src/features/chat/streamEvents.ts`
- Test: `web/src/features/chat/streamEvents.test.ts`

**Interfaces:**
- Consumes: `lib/sse.ts` 的 `parseSSEStream`
- Produces:
  - `type StreamEvent` —— 判别联合：`{kind:"content", text}` / `{kind:"done", status, degraded, stepsExecuted, traceId, sessionId, runId?}` / `{kind:"awaiting_approval", runId, approvals}` / `{kind:"error", message}`
  - `toStreamEvent(raw: Record<string, unknown>): StreamEvent | null`

- [ ] **Step 1: 写失败测试**

`web/src/features/chat/streamEvents.test.ts`：

```ts
import { describe, expect, it } from "vitest";
import { toStreamEvent } from "./streamEvents";

describe("toStreamEvent", () => {
  it("含 content 的是正文片段", () => {
    expect(toStreamEvent({ content: "你", trace_id: "t1" })).toEqual({
      kind: "content",
      text: "你",
    });
  });

  it("awaiting_approval 优先于 done（同一个事件里两者都有）", () => {
    // 后端待审批事件同时带 done:true —— 必须判成审批，否则会把对话当成已结束
    const ev = toStreamEvent({
      awaiting_approval: true,
      run_id: "r1",
      approvals: [{ tool_name: "x" }],
      done: true,
      status: "awaiting_approval",
    });
    expect(ev?.kind).toBe("awaiting_approval");
    if (ev?.kind === "awaiting_approval") {
      expect(ev.runId).toBe("r1");
      expect(ev.approvals).toHaveLength(1);
    }
  });

  it("done 事件带上元数据", () => {
    const ev = toStreamEvent({
      done: true,
      status: "degraded",
      degraded: true,
      steps_executed: 4,
      trace_id: "t9",
      session_id: "s1",
    });
    expect(ev).toEqual({
      kind: "done",
      status: "degraded",
      degraded: true,
      stepsExecuted: 4,
      traceId: "t9",
      sessionId: "s1",
      runId: undefined,
    });
  });

  it("degraded 缺失时按 false 处理", () => {
    const ev = toStreamEvent({ done: true, status: "success" });
    if (ev?.kind === "done") expect(ev.degraded).toBe(false);
  });

  it("error 事件", () => {
    expect(toStreamEvent({ error: "后端炸了" })).toEqual({
      kind: "error",
      message: "后端炸了",
    });
  });

  it("认不出来的事件返回 null", () => {
    expect(toStreamEvent({ 无关字段: 1 })).toBeNull();
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/features/chat/streamEvents.test.ts`
Expected: FAIL，`Failed to resolve import "./streamEvents"`

- [ ] **Step 3: 实现**

`web/src/features/chat/streamEvents.ts`：

```ts
/** 后端 SSE 事件的前端判别联合。 */

export interface ContentEvent {
  kind: "content";
  text: string;
}

export interface DoneEvent {
  kind: "done";
  status: string;
  degraded: boolean;
  stepsExecuted: number;
  traceId: string;
  sessionId: string;
  runId?: string;
}

export interface AwaitingApprovalEvent {
  kind: "awaiting_approval";
  runId: string;
  approvals: unknown[];
}

export interface ErrorEvent {
  kind: "error";
  message: string;
}

export type StreamEvent =
  | ContentEvent
  | DoneEvent
  | AwaitingApprovalEvent
  | ErrorEvent;

function str(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}

/**
 * 把一条原始 SSE 事件转成判别联合。
 *
 * ⚠️ 判定顺序很重要：**先判 awaiting_approval，再判 done**。
 * 后端待审批事件同时带 `done: true`，若先判 done 就会把
 * "图已挂起等审批"误当成"本轮正常结束"，对话会永久卡死。
 */
export function toStreamEvent(
  raw: Record<string, unknown>,
): StreamEvent | null {
  if (raw.error) {
    return { kind: "error", message: str(raw.error) };
  }

  if (raw.awaiting_approval) {
    return {
      kind: "awaiting_approval",
      runId: str(raw.run_id),
      approvals: Array.isArray(raw.approvals) ? raw.approvals : [],
    };
  }

  if (raw.done) {
    return {
      kind: "done",
      status: str(raw.status) || "success",
      degraded: raw.degraded === true,
      stepsExecuted: Number(raw.steps_executed ?? 0) || 0,
      traceId: str(raw.trace_id),
      sessionId: str(raw.session_id),
      runId: raw.run_id ? str(raw.run_id) : undefined,
    };
  }

  if (typeof raw.content === "string" && raw.content.length > 0) {
    return { kind: "content", text: raw.content };
  }

  return null;
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd web && npm test -- --run src/features/chat/streamEvents.test.ts`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add web/src/features/chat
git commit -m "feat(web): SSE 事件判别联合（审批优先于 done）"
```

---

### Task 6: 对话状态机 `useChatStream`（含审批循环）

**Files:**
- Create: `web/src/features/chat/types.ts`
- Create: `web/src/features/chat/useChatStream.ts`
- Test: `web/src/features/chat/useChatStream.test.tsx`

**Interfaces:**
- Consumes: `api/chat.ts` 的 `streamAgentChat`、`api/approvals.ts` 的 `apiResumeRun`、`streamEvents.ts` 的 `toStreamEvent` / `StreamEvent`
- Produces:
  - `types.ts`：`ChatRole = "user" | "assistant"`；`ChatMessage { id, role, text, meta?, interrupted?, error?, approval? }`；`MessageMeta { status, degraded, stepsExecuted, traceId, sessionId }`；`ApprovalRequest { runId, approvals }`
  - `useChatStream(onMessage: (updater: (m: ChatMessage) => ChatMessage) => void): { send, approve, isStreaming, pendingApproval }`

- [ ] **Step 1: 写失败测试（本计划最高价值的一组）**

`web/src/features/chat/useChatStream.test.tsx`：

```tsx
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as approvalsApi from "@/api/approvals";
import * as chatApi from "@/api/chat";
import type { ChatMessage } from "./types";
import { useChatStream } from "./useChatStream";

/** 造一个按顺序吐 SSE 文本的假流。 */
function sseStream(chunks: string[]) {
  const enc = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(c) {
      for (const s of chunks) c.enqueue(enc.encode(s));
      c.close();
    },
  });
}

/** 把消息更新收集进一个可变数组，模拟页面侧的消息列表。 */
function collect() {
  const messages: ChatMessage[] = [];
  const onMessage = (updater: (m: ChatMessage) => ChatMessage) => {
    const idx = messages.length - 1;
    messages[idx] = updater(messages[idx]);
  };
  return { messages, onMessage };
}

afterEach(() => vi.restoreAllMocks());

describe("useChatStream", () => {
  it("累积正文片段并在 done 时写入元数据", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"content":"优先行业"}\n\n',
        'data: {"content":"为金融"}\n\n',
        'data: {"done":true,"status":"success","steps_executed":3,"trace_id":"t1","session_id":"s1"}\n\n',
      ]),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));

    await act(async () => {
      await result.current.send("问题", "s1");
    });

    expect(messages[0]?.text).toBe("优先行业为金融");
    expect(messages[0]?.meta?.traceId).toBe("t1");
    expect(messages[0]?.meta?.stepsExecuted).toBe(3);
    await waitFor(() => expect(result.current.isStreaming).toBe(false));
  });

  it("degraded=true 时元数据被标记", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"content":"部分答案"}\n\n',
        'data: {"done":true,"status":"degraded","degraded":true,"steps_executed":4,"trace_id":"t2","session_id":"s1"}\n\n',
      ]),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });
    expect(messages[0]?.meta?.degraded).toBe(true);
  });

  it("遇到 awaiting_approval 时挂起并暴露 runId，不算结束", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"content":"先做一半"}\n\n',
        'data: {"awaiting_approval":true,"run_id":"r1","approvals":[{"tool_name":"write"}],"done":true,"status":"awaiting_approval"}\n\n',
      ]),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });

    expect(result.current.pendingApproval?.runId).toBe("r1");
    // 已经渲染的内容不能被丢弃
    expect(messages[0]?.text).toBe("先做一半");
    expect(result.current.isStreaming).toBe(false);
  });

  it("批准后从挂起点续流，且可再次遇到 awaiting_approval（循环）", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"awaiting_approval":true,"run_id":"r1","approvals":[{"tool_name":"write"}],"done":true}\n\n',
      ]),
    );
    vi.spyOn(approvalsApi, "apiResumeRun")
      .mockResolvedValueOnce(
        sseStream([
          'data: {"content":"第一次续跑"}\n\n',
          'data: {"awaiting_approval":true,"run_id":"r1","approvals":[{"tool_name":"delete"}],"done":true}\n\n',
        ]),
      )
      .mockResolvedValueOnce(
        sseStream([
          'data: {"content":"最终答案"}\n\n',
          'data: {"done":true,"status":"success","steps_executed":6,"trace_id":"t3","session_id":"s1","run_id":"r1"}\n\n',
        ]),
      );

    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));

    await act(async () => {
      await result.current.send("q", "s1");
    });
    await act(async () => {
      await result.current.approve({ approved: true, comment: "同意" });
    });
    // 第二轮审批仍应挂起，而不是被当成完成
    expect(result.current.pendingApproval?.runId).toBe("r1");

    await act(async () => {
      await result.current.approve({ approved: true, comment: "" });
    });
    expect(result.current.pendingApproval).toBeNull();
    expect(messages[0]?.text).toBe("第一次续跑最终答案");
    expect(messages[0]?.meta?.stepsExecuted).toBe(6);
  });

  it("拒绝时把 approved:false 与 comment 传给后端", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"awaiting_approval":true,"run_id":"r1","approvals":[],"done":true}\n\n',
      ]),
    );
    const resume = vi
      .spyOn(approvalsApi, "apiResumeRun")
      .mockResolvedValue(
        sseStream(['data: {"done":true,"status":"success"}\n\n']),
      );

    const { onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });
    await act(async () => {
      await result.current.approve({ approved: false, comment: "不要写这个文件" });
    });

    expect(resume).toHaveBeenCalledWith(
      "r1",
      { approved: false, comment: "不要写这个文件" },
      expect.anything(),
    );
  });

  it("审批已失效（run_not_found）时给出明确提示且清掉挂起", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream(['data: {"awaiting_approval":true,"run_id":"r1","approvals":[],"done":true}\n\n']),
    );
    vi.spyOn(approvalsApi, "apiResumeRun").mockResolvedValue(
      sseStream(['data: {"error":"检查点不存在","code":"run_not_found"}\n\n']),
    );

    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });
    await act(async () => {
      await result.current.approve({ approved: true, comment: "" });
    });

    expect(result.current.pendingApproval).toBeNull();
    expect(messages[0]?.error).toMatch(/失效|不存在/);
  });

  it("错误事件不破坏已渲染的正文", async () => {
    vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
      sseStream([
        'data: {"content":"已经渲染的部分"}\n\n',
        'data: {"error":"模型超时"}\n\n',
      ]),
    );
    const { messages, onMessage } = collect();
    const { result } = renderHook(() => useChatStream(onMessage));
    await act(async () => {
      await result.current.send("q", "s1");
    });
    expect(messages[0]?.text).toBe("已经渲染的部分");
    expect(messages[0]?.error).toBe("模型超时");
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/features/chat/useChatStream.test.tsx`
Expected: FAIL，`Failed to resolve import "./useChatStream"`

- [ ] **Step 3: 写 `types.ts`**

```ts
export type ChatRole = "user" | "assistant";

export interface MessageMeta {
  status: string;
  degraded: boolean;
  stepsExecuted: number;
  traceId: string;
  sessionId: string;
}

export interface ApprovalRequest {
  runId: string;
  approvals: unknown[];
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  meta?: MessageMeta;
  /** 流式被中断（用户切页/卸载），这一轮没有正常走完 */
  interrupted?: boolean;
  error?: string;
  /** 该助手消息上挂着一个待审批项 */
  approval?: ApprovalRequest;
  /** 该消息对应的审批已失效 */
  approvalExpired?: boolean;
}
```

- [ ] **Step 4: 实现 `useChatStream.ts`**

```ts
import { useCallback, useRef, useState } from "react";
import { apiResumeRun } from "@/api/approvals";
import { streamAgentChat } from "@/api/chat";
import { parseSSEStream } from "@/lib/sse";
import { toStreamEvent } from "./streamEvents";
import type { ApprovalDecision } from "@/api/types";
import type { ApprovalRequest, ChatMessage } from "./types";

export type MessageUpdater = (updater: (m: ChatMessage) => ChatMessage) => void;

/**
 * 对话流状态机。
 *
 * 状态：idle → streaming → (awaiting_approval ⇄ streaming) → done | error
 *
 * ⚠️ 审批是**循环**：续跑中再次命中危险工具会再次推 awaiting_approval
 * （同一个 run_id），因此不能在批准后就把状态清成"结束"。
 *
 * ⚠️ 内容不在这里保管，而是通过 `onMessage` 交给页面的消息列表——
 * 这样用户切页/组件卸载时已渲染内容不会随 hook 一起消失。
 */
export function useChatStream(onMessage: MessageUpdater) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [pendingApproval, setPendingApproval] =
    useState<ApprovalRequest | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  /** 消费一条 SSE 流，把事件作用到当前助手消息上。 */
  const consume = useCallback(
    async (stream: ReadableStream<Uint8Array>) => {
      let sawApproval: ApprovalRequest | null = null;
      let sawError = "";

      for await (const raw of parseSSEStream(stream)) {
        const ev = toStreamEvent(raw);
        if (!ev) continue;

        if (ev.kind === "content") {
          onMessage((m) => ({ ...m, text: m.text + ev.text }));
        } else if (ev.kind === "awaiting_approval") {
          // 记录挂起，等用户决策；**不能**在这里当成结束
          sawApproval = { runId: ev.runId, approvals: ev.approvals };
        } else if (ev.kind === "done") {
          onMessage((m) => ({
            ...m,
            meta: {
              status: ev.status,
              degraded: ev.degraded,
              stepsExecuted: ev.stepsExecuted,
              traceId: ev.traceId,
              sessionId: ev.sessionId,
            },
          }));
        } else if (ev.kind === "error") {
          sawError = ev.message;
        }
      }

      return { sawApproval, sawError };
    },
    [onMessage],
  );

  /** 依据一轮消费结果落定状态。 */
  const settle = useCallback(
    (outcome: { sawApproval: ApprovalRequest | null; sawError: string }) => {
      if (outcome.sawError) {
        const msg = outcome.sawError;
        onMessage((m) => ({
          ...m,
          error: /不存在|检查点/.test(msg) ? "该审批已失效（后端重启或超时）" : msg,
        }));
        setPendingApproval(null);
        setIsStreaming(false);
        return;
      }
      if (outcome.sawApproval) {
        const req = outcome.sawApproval;
        onMessage((m) => ({ ...m, approval: req }));
        setPendingApproval(req);
        setIsStreaming(false);
        return;
      }
      onMessage((m) => ({ ...m, approval: undefined }));
      setPendingApproval(null);
      setIsStreaming(false);
    },
    [onMessage],
  );

  const send = useCallback(
    async (query: string, sessionId: string) => {
      setIsStreaming(true);
      setPendingApproval(null);
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const stream = await streamAgentChat(query, sessionId, controller.signal);
        settle(await consume(stream));
      } catch (e) {
        if (controller.signal.aborted) {
          // 用户主动切走：内容留着，但这一轮标记为已中断，不假装完成
          onMessage((m) => ({ ...m, interrupted: true }));
          setIsStreaming(false);
          return;
        }
        const message = e instanceof Error ? e.message : String(e);
        onMessage((m) => ({ ...m, error: message }));
        setIsStreaming(false);
      }
    },
    [consume, onMessage, settle],
  );

  const approve = useCallback(
    async (decision: ApprovalDecision) => {
      const current = pendingApproval;
      if (!current) return;
      setIsStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const stream = await apiResumeRun(current.runId, decision, controller.signal);
        settle(await consume(stream));
      } catch (e) {
        if (controller.signal.aborted) {
          onMessage((m) => ({ ...m, interrupted: true }));
          setIsStreaming(false);
          return;
        }
        const message = e instanceof Error ? e.message : String(e);
        onMessage((m) => ({ ...m, error: message }));
        setPendingApproval(null);
        setIsStreaming(false);
      }
    },
    [consume, onMessage, pendingApproval, settle],
  );

  const abort = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { send, approve, abort, isStreaming, pendingApproval };
}
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd web && npm test -- --run src/features/chat/useChatStream.test.tsx`
Expected: PASS（7 passed）

- [ ] **Step 6: Commit**

```bash
git add web/src/features/chat
git commit -m "feat(web): 对话流状态机（审批循环 + 失效分支 + 中断保留内容）"
```

---

### Task 7: 对话消息渲染组件

**Files:**
- Create: `web/src/components/chat/Markdown.tsx`
- Create: `web/src/components/chat/DegradedBanner.tsx`
- Create: `web/src/components/chat/MessageMeta.tsx`
- Create: `web/src/components/chat/ApprovalCard.tsx`
- Create: `web/src/components/chat/Message.tsx`
- Create: `web/src/components/chat/Composer.tsx`
- Create: `web/src/components/chat/MessageList.tsx`
- Test: `web/src/components/chat/Message.test.tsx`
- Test: `web/src/components/chat/ApprovalCard.test.tsx`

**Interfaces:**
- Consumes: `features/chat/types.ts` 的 `ChatMessage` / `ApprovalRequest`；`api/types.ts` 的 `ApprovalDecision`
- Produces:
  - `Message({ message, isStreaming, onApprove })`
  - `ApprovalCard({ request, disabled, onDecide })` —— `onDecide(d: ApprovalDecision)`
  - `Composer({ disabled, onSend })` —— `onSend(text: string)`
  - `MessageList({ messages, isStreaming, onApprove })` —— 自动滚动到底

- [ ] **Step 1: 写失败测试**

`web/src/components/chat/DegradedBanner` 的测试放进 `Message.test.tsx`：

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Message } from "./Message";
import type { ChatMessage } from "@/features/chat/types";

function msg(over: Partial<ChatMessage> = {}): ChatMessage {
  return { id: "m1", role: "assistant", text: "回答正文", ...over };
}

describe("Message", () => {
  it("降级时显示显式警示条", () => {
    render(
      <Message
        message={msg({ meta: { status: "degraded", degraded: true, stepsExecuted: 4, traceId: "t", sessionId: "s" } })}
        isStreaming={false}
        onApprove={vi.fn()}
      />,
    );
    expect(screen.getByText(/未取全信息/)).toBeInTheDocument();
  });

  it("成功时只留脚注，不出现警示条", () => {
    render(
      <Message
        message={msg({ meta: { status: "success", degraded: false, stepsExecuted: 3, traceId: "abc123", sessionId: "s" } })}
        isStreaming={false}
        onApprove={vi.fn()}
      />,
    );
    expect(screen.queryByText(/未取全信息/)).not.toBeInTheDocument();
    expect(screen.getByText(/执行 3 步/)).toBeInTheDocument();
    expect(screen.getByText(/abc123/)).toBeInTheDocument();
  });

  it("中断的那一轮被标记，不伪装成正常完成", () => {
    render(<Message message={msg({ interrupted: true })} isStreaming={false} onApprove={vi.fn()} />);
    expect(screen.getByText(/已中断/)).toBeInTheDocument();
  });

  it("错误消息与已有正文同时存在", () => {
    render(<Message message={msg({ error: "模型超时" })} isStreaming={false} onApprove={vi.fn()} />);
    expect(screen.getByText("回答正文")).toBeInTheDocument();
    expect(screen.getByText(/模型超时/)).toBeInTheDocument();
  });

  it("审批失效时给出可理解的说明", () => {
    render(<Message message={msg({ approvalExpired: true })} isStreaming={false} onApprove={vi.fn()} />);
    expect(screen.getByText(/审批已失效/)).toBeInTheDocument();
  });
});
```

`web/src/components/chat/ApprovalCard.test.tsx`：

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApprovalCard } from "./ApprovalCard";

describe("ApprovalCard", () => {
  it("批准时回传 approved:true 与备注", async () => {
    const onDecide = vi.fn();
    render(
      <ApprovalCard
        request={{ runId: "r1", approvals: [{ tool_name: "local_excel_write_tool" }] }}
        disabled={false}
        onDecide={onDecide}
      />,
    );
    await userEvent.type(screen.getByPlaceholderText(/备注/), "同意");
    await userEvent.click(screen.getByRole("button", { name: "批准" }));
    expect(onDecide).toHaveBeenCalledWith({ approved: true, comment: "同意" });
  });

  it("拒绝时回传 approved:false", async () => {
    const onDecide = vi.fn();
    render(
      <ApprovalCard request={{ runId: "r1", approvals: [] }} disabled={false} onDecide={onDecide} />,
    );
    await userEvent.click(screen.getByRole("button", { name: "拒绝" }));
    expect(onDecide).toHaveBeenCalledWith({ approved: false, comment: "" });
  });

  it("处理中时按钮禁用，防止重复提交", () => {
    render(<ApprovalCard request={{ runId: "r1", approvals: [] }} disabled onDecide={vi.fn()} />);
    expect(screen.getByRole("button", { name: "批准" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "拒绝" })).toBeDisabled();
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/components/chat`
Expected: FAIL，`Failed to resolve import "./Message"`

- [ ] **Step 3: 写 `Markdown.tsx`**

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Agent 回答常含表格 / 代码块 / 长路径，统一走 Markdown 渲染。 */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-sm max-w-none break-words">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}
```

- [ ] **Step 4: 写 `DegradedBanner.tsx`**

```tsx
/**
 * 降级警示条。
 *
 * ⚠️ 只应在 `degraded === true` 时渲染。这个字段的语义是
 * "Agent 按现有信息给了诚实的部分答案"——把它藏进一行灰字，
 * 等于让后端那套"诚实部分答案"的机制白做。
 */
export function DegradedBanner() {
  return (
    <div className="mb-3 rounded-md border border-warn-border bg-warn-bg px-3 py-2 text-sm text-warn-text">
      <strong>⚠ 本次回答未取全信息</strong>
      <span className="ml-1">
        —— 部分依据未检索到，结论可能不完整，请勿直接作为决策依据
      </span>
    </div>
  );
}
```

- [ ] **Step 5: 写 `MessageMeta.tsx`**

```tsx
import type { MessageMeta as Meta } from "@/features/chat/types";

/** 消息脚注：执行步数 / 链路追踪 ID / 会话 ID。 */
export function MessageMeta({ meta }: { meta: Meta }) {
  return (
    <div className="mt-2 border-t border-neutral-100 pt-1 text-xs text-neutral-400">
      执行 {meta.stepsExecuted} 步 · trace {meta.traceId || "—"} · session{" "}
      {meta.sessionId || "—"}
    </div>
  );
}
```

- [ ] **Step 6: 写 `ApprovalCard.tsx`**

```tsx
import { useState } from "react";
import type { ApprovalDecision } from "@/api/types";
import type { ApprovalRequest } from "@/features/chat/types";

function describe(a: unknown): string {
  if (!a || typeof a !== "object") return String(a);
  const o = a as Record<string, unknown>;
  const tool = o.tool_name ? String(o.tool_name) : "未知工具";
  const args = o.args_preview ?? o.args;
  return args ? `${tool}：${typeof args === "string" ? args : JSON.stringify(args)}` : tool;
}

export function ApprovalCard({
  request,
  disabled,
  onDecide,
}: {
  request: ApprovalRequest;
  disabled: boolean;
  onDecide: (d: ApprovalDecision) => void;
}) {
  const [comment, setComment] = useState("");
  return (
    <div className="my-3 rounded-md border border-l-4 border-warn-border border-l-warn-border bg-warn-bg p-3">
      <div className="mb-1 font-semibold text-warn-text">⚠ 需要你审批</div>
      <ul className="mb-2 space-y-1 text-sm text-warn-text">
        {request.approvals.map((a, i) => (
          <li key={i} className="rounded bg-white/70 px-2 py-1 font-mono text-xs break-all">
            {describe(a)}
          </li>
        ))}
      </ul>
      <input
        className="mb-2 w-full rounded border border-neutral-300 px-2 py-1 text-sm"
        placeholder="备注（拒绝时会回注给模型）"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        disabled={disabled}
      />
      <div className="flex gap-2">
        <button
          className="rounded bg-blue-600 px-3 py-1 text-sm text-white disabled:opacity-50"
          disabled={disabled}
          onClick={() => onDecide({ approved: true, comment })}
        >
          批准
        </button>
        <button
          className="rounded border border-neutral-300 bg-white px-3 py-1 text-sm disabled:opacity-50"
          disabled={disabled}
          onClick={() => onDecide({ approved: false, comment })}
        >
          拒绝
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `Message.tsx`**

```tsx
import { ApprovalCard } from "./ApprovalCard";
import { DegradedBanner } from "./DegradedBanner";
import { Markdown } from "./Markdown";
import { MessageMeta } from "./MessageMeta";
import type { ApprovalDecision } from "@/api/types";
import type { ChatMessage } from "@/features/chat/types";

export function Message({
  message,
  isStreaming,
  onApprove,
}: {
  message: ChatMessage;
  isStreaming: boolean;
  onApprove: (d: ApprovalDecision) => void;
}) {
  if (message.role === "user") {
    // 用户消息：右对齐浅色块（全宽文档式，非气泡）
    return (
      <div className="mb-4 flex justify-end">
        <div className="max-w-[62%] rounded-xl bg-neutral-100 px-4 py-2 text-sm">
          {message.text}
        </div>
      </div>
    );
  }

  const showCursor = isStreaming && !message.meta && !message.approval && !message.error;

  return (
    <div className="mb-5 flex gap-3">
      <div className="mt-0.5 h-6 w-6 shrink-0 rounded-full bg-blue-600" />
      <div className="min-w-0 flex-1">
        {message.meta?.degraded && <DegradedBanner />}

        {message.text && <Markdown text={message.text} />}
        {showCursor && (
          <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-blue-600 align-text-bottom" />
        )}

        {message.approval && (
          <ApprovalCard
            request={message.approval}
            disabled={isStreaming}
            onDecide={onApprove}
          />
        )}

        {message.approvalExpired && (
          <div className="my-2 rounded border border-neutral-300 bg-neutral-50 px-3 py-2 text-sm text-neutral-600">
            该审批已失效（后端重启或超时），请重新提问。
          </div>
        )}

        {message.interrupted && (
          <div className="mt-2 text-xs text-neutral-400">
            已中断（本轮未走完）
          </div>
        )}

        {message.error && (
          <div className="mt-2 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {message.error}
          </div>
        )}

        {message.meta && <MessageMeta meta={message.meta} />}
      </div>
    </div>
  );
}
```

- [ ] **Step 8: 写 `Composer.tsx` 与 `MessageList.tsx`**

`Composer.tsx`：

```tsx
import { useState } from "react";

export function Composer({
  disabled,
  onSend,
}: {
  disabled: boolean;
  onSend: (text: string) => void;
}) {
  const [text, setText] = useState("");

  function submit() {
    const t = text.trim();
    if (!t || disabled) return;
    setText("");
    onSend(t);
  }

  return (
    <div className="border-t border-neutral-200 px-4 py-3">
      <div className="flex items-end gap-2 rounded-lg border border-neutral-300 px-3 py-2">
        <textarea
          className="max-h-40 flex-1 resize-none text-sm outline-none"
          rows={1}
          placeholder="问点什么…"
          value={text}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            // Enter 发送，Shift+Enter 换行
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button
          className="rounded bg-blue-600 px-3 py-1 text-sm text-white disabled:opacity-40"
          onClick={submit}
          disabled={disabled}
        >
          发送
        </button>
      </div>
      <div className="mt-1 pl-1 text-xs text-neutral-400">Enter 发送 · Shift+Enter 换行</div>
    </div>
  );
}
```

`MessageList.tsx`：

```tsx
import { useEffect, useRef } from "react";
import { Message } from "./Message";
import type { ApprovalDecision } from "@/api/types";
import type { ChatMessage } from "@/features/chat/types";

export function MessageList({
  messages,
  isStreaming,
  onApprove,
}: {
  messages: ChatMessage[];
  isStreaming: boolean;
  onApprove: (d: ApprovalDecision) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // 新内容到达时滚到底（流式回答时持续生效）
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages, isStreaming]);

  return (
    <div className="flex-1 overflow-y-auto px-6 py-5">
      {messages.map((m) => (
        <Message key={m.id} message={m} isStreaming={isStreaming} onApprove={onApprove} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
```

- [ ] **Step 9: 跑测试确认通过**

Run: `cd web && npm test -- --run src/components/chat`
Expected: PASS（8 passed）

- [ ] **Step 10: Commit**

```bash
git add web/src/components/chat
git commit -m "feat(web): 对话消息渲染（降级横幅 / 审批卡片 / 中断与失效标记）"
```

---

### Task 8: 会话本地持久化

**Files:**
- Create: `web/src/features/chat/sessions.ts`
- Test: `web/src/features/chat/sessions.test.ts`

**Interfaces:**
- Consumes: `features/chat/types.ts` 的 `ChatMessage`
- Produces:
  - `interface Session { id: string; title: string; updatedAt: number; messages: ChatMessage[]; pendingRunId?: string }`
  - `listSessions(): Session[]` / `getSession(id): Session | undefined`
  - `createSession(): Session` / `saveSession(s: Session): void` / `deleteSession(id): void`
  - `deriveTitle(text: string): string` —— 前 20 字符，超出加 `…`
  - `newSessionId(): string`

- [ ] **Step 1: 写失败测试**

`web/src/features/chat/sessions.test.ts`：

```ts
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
    const a = { ...createSession(), title: "旧", updatedAt: 1 };
    const b = { ...createSession(), title: "新", updatedAt: 2 };
    saveSession(a);
    saveSession(b);
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/features/chat/sessions.test.ts`
Expected: FAIL，`Failed to resolve import "./sessions"`

- [ ] **Step 3: 实现**

```ts
import type { ChatMessage } from "./types";

/**
 * 会话历史存在前端。
 *
 * ⚠️ 后端**没有**会话列表接口（全项目无 /sessions 之类端点），
 * 它只持有短期记忆。因此"我上次问了什么"只能由前端自己记。
 */
const STORAGE_KEY = "agent-console.sessions";

export interface Session {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
  /** 该会话当前挂起的审批 run_id（用于刷新后回查恢复） */
  pendingRunId?: string;
}

export function newSessionId(): string {
  return crypto.randomUUID();
}

export function createSession(): Session {
  return { id: newSessionId(), title: "新会话", updatedAt: Date.now(), messages: [] };
}

/** 取首条用户消息前 20 个码点作为标题。 */
export function deriveTitle(text: string): string {
  const t = (text ?? "").trim();
  if (!t) return "新会话";
  // 用扩展运算符按码点切，避免把 emoji / 代理对切成半个字符
  const chars = [...t];
  return chars.length <= 20 ? t : chars.slice(0, 20).join("") + "…";
}

function readAll(): Session[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Session[]) : [];
  } catch {
    // 存储被外部写坏时不崩——宁可丢历史，也不要整个页面打不开
    return [];
  }
}

function writeAll(sessions: Session[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {
    // 配额满 / 隐私模式：静默降级为"本次会话不持久化"
  }
}

export function listSessions(): Session[] {
  return readAll().sort((a, b) => b.updatedAt - a.updatedAt);
}

export function getSession(id: string): Session | undefined {
  return readAll().find((s) => s.id === id);
}

export function saveSession(session: Session): void {
  const rest = readAll().filter((s) => s.id !== session.id);
  writeAll([...rest, { ...session, updatedAt: Date.now() }]);
}

export function deleteSession(id: string): void {
  writeAll(readAll().filter((s) => s.id !== id));
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd web && npm test -- --run src/features/chat/sessions.test.ts`
Expected: PASS（10 passed）

- [ ] **Step 5: Commit**

```bash
git add web/src/features/chat/sessions.ts web/src/features/chat/sessions.test.ts
git commit -m "feat(web): 会话本地持久化（标题按码点截断 + 损坏容错）"
```

---

### Task 9: 待审批角标与全局外壳接线

**Files:**
- Create: `web/src/features/approvals/usePendingApprovals.ts`
- Test: `web/src/features/approvals/usePendingApprovals.test.tsx`
- Create: `web/src/components/layout/Sidebar.tsx`
- Create: `web/src/components/layout/AppShell.tsx`
- Test: `web/src/components/layout/Sidebar.test.tsx`
- Create: `web/src/App.tsx`（改为路由表，替换 Task 1 的占位）

**Interfaces:**
- Consumes: `api/approvals.ts` 的 `listPendingApprovals`；`sessions.ts`
- Produces:
  - `usePendingApprovals(): { count: number; items: PendingApproval[]; backend: string }`
    —— TanStack Query，`refetchInterval: 30_000`，`refetchOnWindowFocus: true`
  - `usePendingCountInDocumentTitle(count: number): void` —— 把 `(N) ` 写进 `document.title`
  - `Sidebar({ pendingCount, pendingItems })`
  - `AppShell()` —— 左栏 + `<Outlet/>`，装配轮询与标题

- [ ] **Step 1: 写失败测试（覆盖 Review Focus #5）**

`web/src/features/approvals/usePendingApprovals.test.tsx`：

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api/approvals";
import { usePendingApprovals, usePendingCountInDocumentTitle } from "./usePendingApprovals";

function wrap() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("usePendingApprovals", () => {
  it("返回条数、条目与后端类型", async () => {
    vi.spyOn(api, "listPendingApprovals").mockResolvedValue({
      backend: "redis",
      items: [
        { run_id: "r1", session_id: "s1", tool_name: "write" } as never,
      ],
    });
    const { result } = renderHook(() => usePendingApprovals(), { wrapper: wrap() });
    await waitFor(() => expect(result.current.count).toBe(1));
    expect(result.current.backend).toBe("redis");
  });

  it("接口失败时条数为 0 且不抛（角标不该把页面搞崩）", async () => {
    vi.spyOn(api, "listPendingApprovals").mockRejectedValue(new Error("后端未启动"));
    const { result } = renderHook(() => usePendingApprovals(), { wrapper: wrap() });
    await waitFor(() => expect(result.current.count).toBe(0));
  });
});

describe("usePendingCountInDocumentTitle", () => {
  it("有条数时加前缀，归零时移除", async () => {
    document.title = "Agent 控制台";
    const { rerender } = renderHook(
      ({ n }: { n: number }) => usePendingCountInDocumentTitle(n),
      { initialProps: { n: 0 } },
    );
    expect(document.title).toBe("Agent 控制台");
    rerender({ n: 2 });
    expect(document.title).toBe("(2) Agent 控制台");
    rerender({ n: 0 });
    expect(document.title).toBe("Agent 控制台");
  });
});
```

`web/src/components/layout/Sidebar.test.tsx`：

```tsx
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { Sidebar } from "./Sidebar";

describe("Sidebar", () => {
  it("三个区导航都在", () => {
    render(
      <MemoryRouter>
        <Sidebar pendingCount={0} pendingItems={[]} lower={<div />} />
      </MemoryRouter>,
    );
    expect(screen.getByText(/对话/)).toBeInTheDocument();
    expect(screen.getByText(/文档/)).toBeInTheDocument();
    expect(screen.getByText(/知识库/)).toBeInTheDocument();
  });

  it("有待审批时不显示角标；有条数时显示数字", () => {
    const { rerender } = render(
      <MemoryRouter>
        <Sidebar pendingCount={0} pendingItems={[]} lower={<div />} />
      </MemoryRouter>,
    );
    expect(screen.queryByText("待审批")).not.toBeInTheDocument();

    rerender(
      <MemoryRouter>
        <Sidebar
          pendingCount={2}
          pendingItems={[{ run_id: "r1", session_id: "s1", query: "问一句" } as never]}
          lower={<div />}
        />
      </MemoryRouter>,
    );
    expect(screen.getByText("待审批")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("内存后端时如实提示重启会失效", () => {
    render(
      <MemoryRouter>
        <Sidebar pendingCount={1} pendingItems={[]} backend="memory" lower={<div />} />
      </MemoryRouter>,
    );
    expect(screen.getByText(/后端重启/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/features/approvals src/components/layout`
Expected: FAIL，模块不存在

- [ ] **Step 3: 实现 `usePendingApprovals.ts`**

```ts
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
```

- [ ] **Step 4: 实现 `Sidebar.tsx`**

```tsx
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import type { PendingApproval } from "@/api/types";

const NAV = [
  { to: "/", label: "💬 对话" },
  { to: "/documents", label: "📄 文档" },
  { to: "/knowledgebase", label: "🕸 知识库" },
];

export function Sidebar({
  pendingCount,
  pendingItems,
  backend,
  lower,
}: {
  pendingCount: number;
  pendingItems: PendingApproval[];
  backend?: string;
  /** 下半栏内容：随当前区切换（会话列表 / RAG 集合 / 图谱集合） */
  lower: ReactNode;
}) {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-neutral-200 bg-neutral-50">
      <nav className="p-2">
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.to === "/"}
            className={({ isActive }) =>
              `block rounded px-3 py-2 text-sm ${
                isActive ? "bg-neutral-200 font-semibold" : "text-neutral-600 hover:bg-neutral-100"
              }`
            }
          >
            {n.label}
          </NavLink>
        ))}
      </nav>

      <div className="min-h-0 flex-1 overflow-y-auto border-t border-dashed border-neutral-300 px-2 py-2">
        {lower}
      </div>

      {pendingCount > 0 && (
        <div className="border-t border-neutral-200 p-2">
          <div className="rounded border border-warn-border bg-warn-bg px-2 py-1.5 text-xs font-semibold text-warn-text">
            ⚠ 待审批
            <span className="ml-1 rounded-full bg-warn-text px-1.5 text-white">
              {pendingCount}
            </span>
          </div>
          <div className="mt-1 truncate text-[10px] text-neutral-400">
            {pendingItems[0]?.query ?? pendingItems[0]?.tool_name ?? "—"}
          </div>
          {backend === "memory" && (
            <div className="mt-1 text-[10px] text-warn-text">
              当前为内存模式，后端重启后将失效
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
```

- [ ] **Step 5: 实现 `AppShell.tsx`（装配轮询、标题、下半栏）**

```tsx
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { usePendingApprovals, usePendingCountInDocumentTitle } from "@/features/approvals/usePendingApprovals";
import { useSessionList } from "@/features/chat/useSessionList"; // 见 Task 10 Step 3
import { useKbCollectionList } from "@/features/documents/useKbCollectionList"; // 见 Task 10 Step 3
import { useGraphCollectionList } from "@/features/knowledgebase/useGraphCollectionList"; // 见 Task 11 Step 3

export function AppShell() {
  const { count, items, backend } = usePendingApprovals();
  usePendingCountInDocumentTitle(count);
  const { pathname } = useLocation();

  // 左栏下半栏：内容随当前区切换
  const sessions = useSessionList();
  const kbCollections = useKbCollectionList();
  const graphCollections = useGraphCollectionList();

  const lower = pathname.startsWith("/documents") ? (
    <CollectionListPane title="RAG 集合" items={kbCollections.data ?? []} base="/documents" />
  ) : pathname.startsWith("/knowledgebase") ? (
    <CollectionListPane title="图谱集合" items={graphCollections.data ?? []} base="/knowledgebase" />
  ) : (
    <SessionListPane sessions={sessions} />
  );

  return (
    <div className="flex h-full">
      <Sidebar pendingCount={count} pendingItems={items} backend={backend} lower={lower} />
      <main className="min-w-0 flex-1">
        <Outlet />
      </main>
    </div>
  );
}
```

`CollectionListPane` 与 `SessionListPane` 直接写在同文件内（各约 20 行），
前者渲染 `NavLink` 到 `${base}/${name}`，后者渲染会话标题列表并高亮当前会话。

⚠️ **`useKbCollectionList` / `useGraphCollectionList` 必须用固定 query key**
（`["kb-collections"]` / `["graph-collections"]`），切区时共用缓存而不重新请求。

- [ ] **Step 6: 改 `App.tsx` 为路由表**

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { ChatPage } from "@/features/chat/ChatPage";
import { DocumentsPage } from "@/features/documents/DocumentsPage";
import { KnowledgeBasePage } from "@/features/knowledgebase/KnowledgeBasePage";

const queryClient = new QueryClient();

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <ChatPage /> },
      { path: "documents", element: <DocumentsPage /> },
      { path: "documents/:collectionName", element: <DocumentsPage /> },
      { path: "knowledgebase", element: <KnowledgeBasePage /> },
      { path: "knowledgebase/:collectionName", element: <KnowledgeBasePage /> },
    ],
  },
]);

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
```

- [ ] **Step 7: 跑测试确认通过**

Run: `cd web && npm test -- --run src/features/approvals src/components/layout`
Expected: PASS（5 passed）

- [ ] **Step 8: Commit**

```bash
git add web/src
git commit -m "feat(web): 应用外壳、路由与待审批角标（30s 轮询 + 标签页标题）"
```

---

### Task 10: 文档区（RAG 集合管理）

**Files:**
- Create: `web/src/features/documents/useKbCollectionList.ts`
- Create: `web/src/features/chat/useSessionList.ts`
- Create: `web/src/features/documents/DocumentsPage.tsx`
- Test: `web/src/features/documents/DocumentsPage.test.tsx`

**Interfaces:**
- Consumes: `api/documents.ts` 全部函数；`components/ui`（shadcn 的 `Button` / `Table` / `Dialog` / `toast`）
- Produces: `DocumentsPage()` —— 无集合选中时显示总览 + 上传入口；有 `:collectionName` 时显示该集合的文件表

- [ ] **Step 1: 写失败测试（覆盖 Review Focus #3 文件名编码）**

`web/src/features/documents/DocumentsPage.test.tsx`：

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api/documents";
import { DocumentsPage } from "./DocumentsPage";

function renderAt(path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/documents/:collectionName" element={<DocumentsPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => vi.restoreAllMocks());

describe("DocumentsPage", () => {
  it("列出 RAG 集合", async () => {
    vi.spyOn(api, "listKbCollections").mockResolvedValue([
      { name: "sales_kb", description: "销售知识", document_count: 3 },
    ]);
    renderAt("/documents");
    await waitFor(() => expect(screen.getByText("sales_kb")).toBeInTheDocument());
  });

  it("显示集合内文件与切片数", async () => {
    vi.spyOn(api, "getKbCollectionFiles").mockResolvedValue({
      name: "sales_kb",
      description: "销售知识",
      document_count: 2,
      files: [
        { filename: "年度目标.xlsx", chunk_count: 42 },
        { filename: "折扣权限与审批规则.md", chunk_count: 11 },
      ],
    });
    renderAt("/documents/sales_kb");
    await waitFor(() =>
      expect(screen.getByText("折扣权限与审批规则.md")).toBeInTheDocument(),
    );
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("删除中文文件名时做 URL 编码", async () => {
    // ⚠️ Review Focus #3：不编码会 404，且报错含糊
    vi.spyOn(api, "getKbCollectionFiles").mockResolvedValue({
      name: "sales_kb",
      description: null,
      document_count: 1,
      files: [{ filename: "线索阶段流转规则.md", chunk_count: 1 }],
    });
    const del = vi
      .spyOn(api, "deleteKbCollectionFile")
      .mockResolvedValue({} as never);

    renderAt("/documents/sales_kb");
    await waitFor(() => screen.getByText("线索阶段流转规则.md"));
    await userEvent.click(screen.getAllByRole("button", { name: "删除" })[0]);
    await userEvent.click(screen.getByRole("button", { name: "确认删除" }));

    expect(del).toHaveBeenCalledWith("sales_kb", "线索阶段流转规则.md");
  });

  it("上传成功后刷新文件表", async () => {
    const files = vi.spyOn(api, "getKbCollectionFiles").mockResolvedValue({
      name: "sales_kb",
      description: null,
      document_count: 0,
      files: [],
    });
    vi.spyOn(api, "uploadDocument").mockResolvedValue({} as never);

    renderAt("/documents/sales_kb");
    await waitFor(() => expect(files).toHaveBeenCalled());

    const file = new File(["x"], "新规则.md", { type: "text/markdown" });
    await userEvent.upload(screen.getByLabelText(/选择文件/), file);
    await userEvent.click(screen.getByRole("button", { name: "上传" }));

    await waitFor(() => expect(api.uploadDocument).toHaveBeenCalled());
    // 上传后必须重新拉取，否则用户看不到刚传的文件
    await waitFor(() =>
      expect(files.mock.calls.length).toBeGreaterThan(1),
    );
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/features/documents`
Expected: FAIL，模块不存在

- [ ] **Step 3: 写两个列表 hook**

`web/src/features/documents/useKbCollectionList.ts`：

```ts
import { useQuery } from "@tanstack/react-query";
import { listKbCollections } from "@/api/documents";

/** ⚠️ 固定 query key：左栏与页面共用同一份缓存，切区不重新请求。 */
export function useKbCollectionList() {
  return useQuery({
    queryKey: ["kb-collections"],
    queryFn: listKbCollections,
    retry: false,
  });
}
```

`web/src/features/chat/useSessionList.ts`：

```ts
import { useCallback, useState } from "react";
import { listSessions, saveSession, type Session } from "./sessions";

/** 会话列表：本地存储，不需要 TanStack Query。 */
export function useSessionList(): {
  sessions: Session[];
  refresh: () => void;
} {
  const [sessions, setSessions] = useState<Session[]>(() => listSessions());
  const refresh = useCallback(() => setSessions(listSessions()), []);
  return { sessions, refresh };
}

export { saveSession };
```

- [ ] **Step 4: 实现 `DocumentsPage.tsx`**

```tsx
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import {
  deleteKbCollectionFile,
  getKbCollectionFiles,
  listDocuments,
  uploadDocument,
} from "@/api/documents";
import { useKbCollectionList } from "./useKbCollectionList";

export function DocumentsPage() {
  const { collectionName } = useParams();
  const qc = useQueryClient();
  const collections = useKbCollectionList();

  const files = useQuery({
    queryKey: ["kb-files", collectionName],
    queryFn: () => getKbCollectionFiles(collectionName!),
    enabled: Boolean(collectionName),
    retry: false,
  });

  const allDocs = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
    enabled: !collectionName,
    retry: false,
  });

  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");

  const upload = useMutation({
    mutationFn: () => uploadDocument(pendingFile!, collectionName!, description),
    onSuccess: () => {
      setPendingFile(null);
      setDescription("");
      // 上传后必须失效缓存，否则文件表看不到刚传的东西
      void qc.invalidateQueries({ queryKey: ["kb-files", collectionName] });
      void qc.invalidateQueries({ queryKey: ["kb-collections"] });
      void qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  const remove = useMutation({
    mutationFn: (filename: string) =>
      deleteKbCollectionFile(collectionName!, filename),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["kb-files", collectionName] });
      void qc.invalidateQueries({ queryKey: ["kb-collections"] });
    },
  });

  if (!collectionName) {
    return (
      <div className="p-6">
        <h1 className="mb-4 text-lg font-semibold">文档</h1>
        <p className="mb-4 text-sm text-neutral-500">
          从左侧选择一个集合查看其文件，或直接在下方查看全部文档。
        </p>
        <table className="w-full text-sm">
          <thead className="bg-neutral-50 text-left text-neutral-500">
            <tr>
              <th className="px-3 py-2">文件</th>
              <th className="px-3 py-2">集合</th>
            </tr>
          </thead>
          <tbody>
            {(allDocs.data ?? []).map((d) => (
              <tr key={d.document_id} className="border-t border-neutral-100">
                <td className="px-3 py-2">{d.filename}</td>
                <td className="px-3 py-2 text-neutral-500">{d.collection_name}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold">{collectionName}</h1>
      <p className="mb-4 text-sm text-neutral-500">
        {files.data?.description ?? "（该集合暂无描述）"}
      </p>

      <div className="mb-5 rounded border border-neutral-200 p-3">
        <label className="block text-sm font-medium" htmlFor="kb-file">
          选择文件
        </label>
        <input
          id="kb-file"
          type="file"
          className="mt-1 block text-sm"
          onChange={(e) => setPendingFile(e.target.files?.[0] ?? null)}
        />
        <input
          className="mt-2 w-full rounded border border-neutral-300 px-2 py-1 text-sm"
          placeholder="集合描述（留空则不更新；这个描述会影响后续意图路由）"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
        <button
          className="mt-2 rounded bg-blue-600 px-3 py-1 text-sm text-white disabled:opacity-40"
          disabled={!pendingFile || upload.isPending}
          onClick={() => upload.mutate()}
        >
          上传
        </button>
        {upload.isError && (
          <div className="mt-2 text-sm text-red-600">
            上传失败：{(upload.error as Error).message}
          </div>
        )}
      </div>

      {files.isError && (
        <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          加载失败：{(files.error as Error).message}
          <button className="ml-2 underline" onClick={() => void files.refetch()}>
            重试
          </button>
        </div>
      )}

      <table className="w-full text-sm">
        <thead className="bg-neutral-50 text-left text-neutral-500">
          <tr>
            <th className="px-3 py-2">文件</th>
            <th className="px-3 py-2">切片</th>
            <th className="px-3 py-2" />
          </tr>
        </thead>
        <tbody>
          {(files.data?.files ?? []).map((f) => (
            <tr key={f.filename} className="border-t border-neutral-100">
              <td className="px-3 py-2">{f.filename}</td>
              <td className="px-3 py-2 text-neutral-500">
                {f.chunk_count ?? f.chunks ?? "—"}
              </td>
              <td className="px-3 py-2 text-right">
                <button
                  className="text-neutral-500 underline"
                  onClick={() => {
                    // 二次确认：删除是破坏性操作且不可撤销
                    if (window.confirm(`确认删除 ${f.filename}？`)) {
                      remove.mutate(f.filename);
                    }
                  }}
                >
                  删除
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

⚠️ 测试里的"确认删除"按钮对应的是 shadcn `Dialog` 版本；
若先用 `window.confirm`，测试改用 `vi.spyOn(window, "confirm").mockReturnValue(true)`。
**两种都行，但要在实现时选定一种并让测试与之匹配**，不要两套并存。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd web && npm test -- --run src/features/documents`
Expected: PASS（4 passed）

- [ ] **Step 6: Commit**

```bash
git add web/src/features/documents web/src/features/chat/useSessionList.ts
git commit -m "feat(web): 文档区（RAG 集合文件管理 + 上传 + 删除）"
```

---

### Task 11: 知识库区（图谱集合，独立定制）

**Files:**
- Create: `web/src/features/knowledgebase/useGraphCollectionList.ts`
- Create: `web/src/features/knowledgebase/KnowledgeBasePage.tsx`
- Test: `web/src/features/knowledgebase/KnowledgeBasePage.test.tsx`

**Interfaces:**
- Consumes: `api/knowledgebase.ts` 全部函数
- Produces: `KnowledgeBasePage()` —— 以"图谱建得怎么样"为中心：统计 + 处理状态表 + 单文件上传 + 维护入口

- [ ] **Step 1: 写失败测试**

`web/src/features/knowledgebase/KnowledgeBasePage.test.tsx`：

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api/knowledgebase";
import { KnowledgeBasePage } from "./KnowledgeBasePage";

function renderAt(path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/knowledgebase" element={<KnowledgeBasePage />} />
          <Route path="/knowledgebase/:collectionName" element={<KnowledgeBasePage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => vi.restoreAllMocks());

describe("KnowledgeBasePage", () => {
  it("列出图谱集合", async () => {
    vi.spyOn(api, "listGraphCollections").mockResolvedValue([
      { name: "Cross_Entity_Relation", legacy: false, description: "跨实体", document_count: 2, files: [] },
    ]);
    renderAt("/knowledgebase");
    await waitFor(() =>
      expect(screen.getByText("Cross_Entity_Relation")).toBeInTheDocument(),
    );
  });

  it("显示每个文件的处理状态", async () => {
    vi.spyOn(api, "getGraphCollectionFiles").mockResolvedValue({
      name: "Cross_Entity_Relation",
      legacy: false,
      description: "跨实体",
      document_count: 2,
      files: [
        { filename: "系统关系说明.pdf", status: "已处理", chunk_count: 86 },
        { filename: "组织架构.txt", status: "处理中", chunk_count: 0 },
      ],
    });
    renderAt("/knowledgebase/Cross_Entity_Relation");
    await waitFor(() => expect(screen.getByText("已处理")).toBeInTheDocument());
    expect(screen.getByText("处理中")).toBeInTheDocument();
  });

  it("如实说明无法查询进度（后端没有任务查询接口）", async () => {
    vi.spyOn(api, "getGraphCollectionFiles").mockResolvedValue({
      name: "c", legacy: false, description: null, document_count: 0, files: [],
    });
    renderAt("/knowledgebase/c");
    await waitFor(() =>
      expect(screen.getByText(/无法查询进度/)).toBeInTheDocument(),
    );
  });

  it("上传走单文件同步接口，不用批量接口", async () => {
    vi.spyOn(api, "getGraphCollectionFiles").mockResolvedValue({
      name: "c", legacy: false, description: null, document_count: 0, files: [],
    });
    const up = vi.spyOn(api, "uploadGraphDocument").mockResolvedValue({} as never);
    renderAt("/knowledgebase/c");
    await waitFor(() => screen.getByLabelText(/选择文件/));

    const file = new File(["x"], "a.pdf", { type: "application/pdf" });
    await (await import("@testing-library/user-event")).default.upload(
      screen.getByLabelText(/选择文件/),
      file,
    );
    await (await import("@testing-library/user-event")).default.click(
      screen.getByRole("button", { name: "上传" }),
    );
    await waitFor(() => expect(up).toHaveBeenCalled());
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd web && npm test -- --run src/features/knowledgebase`
Expected: FAIL，模块不存在

- [ ] **Step 3: 写 `useGraphCollectionList.ts`**

```ts
import { useQuery } from "@tanstack/react-query";
import { listGraphCollections } from "@/api/knowledgebase";

/** ⚠️ 固定 query key：左栏与页面共用缓存。 */
export function useGraphCollectionList() {
  return useQuery({
    queryKey: ["graph-collections"],
    queryFn: listGraphCollections,
    retry: false,
  });
}
```

- [ ] **Step 4: 实现 `KnowledgeBasePage.tsx`**

以"图谱建得怎么样"为中心（**不与文档区共用组件**——用户已决定，
因为图谱后续要单独加功能）：

```tsx
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import {
  clearLegacyWorkspace,
  deleteGraphCollectionFile,
  getGraphCollectionFiles,
  uploadGraphDocument,
} from "@/api/knowledgebase";
import { useGraphCollectionList } from "./useGraphCollectionList";

const STATUS_STYLE: Record<string, string> = {
  已处理: "text-green-600",
  处理中: "text-amber-600",
  失败: "text-red-600",
};

export function KnowledgeBasePage() {
  const { collectionName } = useParams();
  const qc = useQueryClient();
  const collections = useGraphCollectionList();

  const detail = useQuery({
    queryKey: ["graph-files", collectionName],
    queryFn: () => getGraphCollectionFiles(collectionName!),
    enabled: Boolean(collectionName),
    retry: false,
  });

  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");

  const upload = useMutation({
    // ⚠️ 单文件同步接口（/documents/kownledgebase/upload，注意拼写）。
    // 批量接口返回 202，但后端没有任务查询端点，用了就只能"提交后刷新"。
    mutationFn: () => uploadGraphDocument(pendingFile!, collectionName!, description),
    onSuccess: () => {
      setPendingFile(null);
      setDescription("");
      void qc.invalidateQueries({ queryKey: ["graph-files", collectionName] });
      void qc.invalidateQueries({ queryKey: ["graph-collections"] });
    },
  });

  const remove = useMutation({
    mutationFn: (filename: string) =>
      deleteGraphCollectionFile(collectionName!, filename),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["graph-files", collectionName] });
      void qc.invalidateQueries({ queryKey: ["graph-collections"] });
    },
  });

  const clearLegacy = useMutation({
    mutationFn: clearLegacyWorkspace,
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["graph-collections"] }),
  });

  if (!collectionName) {
    return (
      <div className="p-6">
        <h1 className="mb-4 text-lg font-semibold">知识库</h1>
        <p className="mb-4 text-sm text-neutral-500">
          从左侧选择一个图谱集合。图谱抽取在后台进行，构建完成后实体与关系才可用于检索。
        </p>
        <table className="w-full text-sm">
          <thead className="bg-neutral-50 text-left text-neutral-500">
            <tr>
              <th className="px-3 py-2">集合</th>
              <th className="px-3 py-2">描述</th>
              <th className="px-3 py-2">文件</th>
            </tr>
          </thead>
          <tbody>
            {(collections.data ?? []).map((c) => (
              <tr key={c.name} className="border-t border-neutral-100">
                <td className="px-3 py-2">{c.name}</td>
                <td className="px-3 py-2 text-neutral-500">{c.description ?? "—"}</td>
                <td className="px-3 py-2 text-neutral-500">{c.document_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <button
          className="mt-4 text-sm text-neutral-500 underline"
          onClick={() => {
            if (window.confirm("清空历史遗留 workspace？具名集合不受影响。")) {
              clearLegacy.mutate();
            }
          }}
        >
          清空历史遗留 workspace
        </button>
      </div>
    );
  }

  const totalChunks = (detail.data?.files ?? []).reduce(
    (n, f) => n + (f.chunk_count ?? 0),
    0,
  );

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold">{collectionName}</h1>
      <p className="mb-3 text-sm text-neutral-500">
        {detail.data?.description ?? "（该集合暂无描述）"}
      </p>

      <div className="mb-5 flex gap-2 text-sm">
        <span className="rounded bg-neutral-100 px-3 py-1">
          文件 <strong>{detail.data?.document_count ?? 0}</strong>
        </span>
        <span className="rounded bg-neutral-100 px-3 py-1">
          切片 <strong>{totalChunks}</strong>
        </span>
      </div>

      <div className="mb-5 rounded border border-neutral-200 p-3">
        <label className="block text-sm font-medium" htmlFor="graph-file">
          选择文件
        </label>
        <input
          id="graph-file"
          type="file"
          accept=".pdf,.txt"
          className="mt-1 block text-sm"
          onChange={(e) => setPendingFile(e.target.files?.[0] ?? null)}
        />
        <input
          className="mt-2 w-full rounded border border-neutral-300 px-2 py-1 text-sm"
          placeholder="集合描述（留空则不更新）"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
        <button
          className="mt-2 rounded bg-blue-600 px-3 py-1 text-sm text-white disabled:opacity-40"
          disabled={!pendingFile || upload.isPending}
          onClick={() => upload.mutate()}
        >
          上传
        </button>
        {upload.isPending && (
          <span className="ml-2 text-sm text-neutral-500">正在解析并织入图谱…</span>
        )}
        {upload.isError && (
          <div className="mt-2 text-sm text-red-600">
            上传失败：{(upload.error as Error).message}
          </div>
        )}
        <div className="mt-2 rounded bg-neutral-50 px-3 py-2 text-xs text-neutral-500">
          图谱抽取在后台进行。<strong>提交后无法查询进度</strong>
          （后端未提供任务查询接口），请稍后回来刷新本页。
        </div>
      </div>

      <table className="w-full text-sm">
        <thead className="bg-neutral-50 text-left text-neutral-500">
          <tr>
            <th className="px-3 py-2">文件</th>
            <th className="px-3 py-2">处理状态</th>
            <th className="px-3 py-2">切片</th>
            <th className="px-3 py-2" />
          </tr>
        </thead>
        <tbody>
          {(detail.data?.files ?? []).map((f) => (
            <tr key={f.filename} className="border-t border-neutral-100">
              <td className="px-3 py-2">{f.filename}</td>
              <td className={`px-3 py-2 ${STATUS_STYLE[f.status ?? ""] ?? "text-neutral-500"}`}>
                {f.status ?? "—"}
              </td>
              <td className="px-3 py-2 text-neutral-500">{f.chunk_count ?? "—"}</td>
              <td className="px-3 py-2 text-right">
                <button
                  className="text-neutral-500 underline"
                  onClick={() => {
                    if (window.confirm(`确认删除 ${f.filename}？`)) {
                      remove.mutate(f.filename);
                    }
                  }}
                >
                  删除
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd web && npm test -- --run src/features/knowledgebase`
Expected: PASS（4 passed）

- [ ] **Step 6: Commit**

```bash
git add web/src/features/knowledgebase
git commit -m "feat(web): 知识库区（图谱集合统计 + 处理状态 + 单文件同步上传）"
```

---

### Task 12: 后端静态挂载（唯一的后端改动）

**Files:**
- Modify: `app/main.py`（在 router 挂载之后增加静态与 SPA 回退）
- Test: `测试/test_static_frontend_mount.py`

**Interfaces:**
- Consumes: 无（改动独立）
- Produces: 访问 `/` 返回 `web/dist/index.html`；`/api/v1/*` 行为不变

- [ ] **Step 1: 写失败测试**

`测试/test_static_frontend_mount.py`：

```python
# -*- coding: utf-8 -*-
"""后端托管前端构建产物的行为约定。

要点（见设计文档 4.1 / 第 9 节）：
  - 非 /api/ 前缀的未知路径必须回退到 index.html，否则前端路由一刷新就 404
  - /api/ 前缀一律交给 router，不得被静态挂载拦截
  - dist 不存在时后端必须照常启动（不能因为前端没构建就起不来）
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import _mount_frontend  # noqa: PLC0415


def _make_dist(tmp_path: Path, html: str = "<html><body>SPA</body></html>") -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(html, encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    return dist


def test_mount_serves_index_at_root(tmp_path: Path) -> None:
    app = FastAPI()
    _mount_frontend(app, _make_dist(tmp_path))
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "SPA" in res.text


def test_deep_frontend_route_falls_back_to_index(tmp_path: Path) -> None:
    """⚠️ Review Focus：/documents 这类前端路由刷新必须不 404。"""
    app = FastAPI()
    _mount_frontend(app, _make_dist(tmp_path))
    client = TestClient(app)
    res = client.get("/documents/sales_kb")
    assert res.status_code == 200
    assert "SPA" in res.text


def test_static_asset_is_served(tmp_path: Path) -> None:
    app = FastAPI()
    dist = _make_dist(tmp_path)
    _mount_frontend(app, dist)
    client = TestClient(app)
    res = client.get("/assets/app.js")
    assert res.status_code == 200
    assert "console.log" in res.text


def test_api_paths_are_not_intercepted(tmp_path: Path) -> None:
    """⚠️ /api/ 前缀必须原样交给 router——被静态挂载吃掉就是灾难。"""
    app = FastAPI()

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    _mount_frontend(app, _make_dist(tmp_path))
    client = TestClient(app)
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_unknown_api_path_returns_404_not_index(tmp_path: Path) -> None:
    """不存在的 /api/ 路径要 404，不能悄悄返回 HTML——那会让前端解析 JSON 时炸。"""
    app = FastAPI()
    _mount_frontend(app, _make_dist(tmp_path))
    client = TestClient(app)
    res = client.get("/api/v1/definitely-not-here")
    assert res.status_code == 404


def test_missing_dist_does_not_break_startup(tmp_path: Path) -> None:
    app = FastAPI()
    _mount_frontend(app, tmp_path / "does-not-exist")  # 不应抛错
    client = TestClient(app)
    assert client.get("/").status_code == 404
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& "d:/pycharm/PyCharm 2026.1.1/PythonProject_deepagents/.venv/Scripts/python.exe" -m pytest 测试/test_static_frontend_mount.py -q`
Expected: FAIL，`ImportError: cannot import name '_mount_frontend' from 'app.main'`

- [ ] **Step 3: 实现（在 `app/main.py` 的 router 挂载之后追加）**

```python
# ── 前端静态托管（单源，免 CORS）──────────────────────────────────────────
# 顺序无关紧要（FastAPI 按注册顺序匹配），但**回退路由必须最后注册**，
# 否则它会抢在 router 前面把 /api/ 请求吃掉。
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def _mount_frontend(application: FastAPI, dist_dir: Path) -> None:
    """把前端构建产物挂到应用上；``dist_dir`` 不存在时安静跳过。

    设计要点（见 docs/superpowers/specs/2026-09-19-agent-console-frontend-design.md 4.1）：
      - 真实静态文件走 StaticFiles
      - **非 /api/ 前缀**的未知路径回退 index.html（前端路由刷新不 404）
      - **/api/ 前缀一律 404**，绝不能被回退逻辑吞掉——否则前端会把 HTML 当 JSON 解析
      - dist 不存在时只告警，不影响后端启动（前端尚未构建也能起服务）
    """
    index_file = dist_dir / "index.html"
    if not index_file.is_file():
        logger.warning(
            "未找到前端构建产物 {}，跳过静态挂载（后端照常启动）。"
            "构建命令：cd web && npm run build",
            index_file,
        )
        return

    application.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="assets")

    @application.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            # 交给 FastAPI 的正常 404，不要用 index.html 冒充
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = dist_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_file)


_mount_frontend(application, Path(__file__).resolve().parents[1] / "web" / "dist")
```

⚠️ `Path` / `logger` / `HTTPException` 若在 `app/main.py` 中尚未导入，一并补上：
`from pathlib import Path`、`from fastapi import HTTPException`（`logger` 该文件已有）。

- [ ] **Step 4: 跑新测试确认通过**

Run: `& "d:/pycharm/PyCharm 2026.1.1/PythonProject_deepagents/.venv/Scripts/python.exe" -m pytest 测试/test_static_frontend_mount.py -q`
Expected: PASS（6 passed）

- [ ] **Step 5: 跑后端全量回归**

Run: `& "d:/pycharm/PyCharm 2026.1.1/PythonProject_deepagents/.venv/Scripts/python.exe" -m pytest evals 测试 benchmark -q`
Expected: **376 passed, 1 failed**（那 1 项是既有失败，与本变更无关；数量不得下降）

- [ ] **Step 6: Commit**

```bash
git add app/main.py 测试/test_static_frontend_mount.py
git commit -m "feat(api): 挂载前端构建产物（SPA 回退且不拦截 /api/，缺 dist 时照常启动）"
```

---

### Task 13: 端到端验收

**Files:**
- Create: `web/README.md`（运行说明）
- Modify: `README.md`（仓库根，补前端一节）

**Interfaces:**
- Consumes: 全部前置任务
- Produces: 可交付运行的整套系统

- [ ] **Step 1: 构建前端**

```bash
cd web && npm run build
```

Expected: 产出 `web/dist/index.html` 与 `web/dist/assets/*`

- [ ] **Step 2: 跑全部前端测试**

Run: `cd web && npm test -- --run`
Expected: 全部 PASS（本计划累计约 44 项）

- [ ] **Step 3: 起后端并走一遍真实对话**

```bash
uvicorn app.main:app
```

浏览器打开 `http://127.0.0.1:8000/`，然后：

1. 输入 `我们优先做哪些行业？哪些算次优先？` → 应看到流式回答逐字出现
2. 回答结束后检查脚注是否出现（`执行 N 步 · trace … · session …`）
3. 若出现黄色审批卡片 → 点"批准" → 应继续流式输出（**不是**重新开始）
4. 若该轮 `degraded=true` → 应出现黄色警示条

- [ ] **Step 4: 验收待审批角标（关键路径）**

在对话触发审批后，**不要处理**，直接点左栏"文档"：

- 左栏底部应出现"⚠ 待审批 (1)"
- 浏览器标签页标题应变成 `(1) Agent 控制台`
- **刷新页面** → 切回对话 → 审批卡片应**自动重建**（靠 localStorage 里的 `run_id` 回查）

- [ ] **Step 5: 验收两个管理区**

1. 文档区：选一个集合 → 看到文件表 → 上传一个小 `.md` → 文件表应刷新出现它 → 删除它
2. 知识库区：选一个图谱集合 → 看到处理状态列 → 上传一个小 `.txt` → 出现"正在解析并织入图谱…"

**中文文件名必须试**（上传一个含中文名的文件再删），这是 Review Focus #3 的实测。

- [ ] **Step 6: 写运行说明**

`web/README.md`：

```markdown
# Agent 控制台前端

## 开发

```bash
npm install
npm run dev     # http://localhost:5173，/api 经 vite proxy 转发到 127.0.0.1:8000
npm test        # vitest
```

后端需同时运行：`uvicorn app.main:app --reload`

## 日常使用（单进程、同源、无 CORS）

```bash
npm run build   # 产出 dist/
# 然后只起后端：uvicorn app.main:app
# 访问 http://127.0.0.1:8000/
```

## 注意

- 是 **Tailwind v4**：没有 `tailwind.config.js`，主题写在 `src/index.css` 的 `@theme {}` 里
- 图谱单文件上传的路径是 `kownledgebase`（后端实际拼写），**不要改成正确拼写**
- 会话历史存在浏览器 localStorage（后端没有会话列表接口）
```

仓库根 `README.md` 补一节"前端控制台"，指向 `web/README.md`。

- [ ] **Step 7: Commit**

```bash
git add web/README.md README.md
git commit -m "docs: 前端控制台运行说明与端到端验收记录"
```

---

## 自审记录（写完计划后按规格复查）

按 writing-plans 的四步自审执行，结果如下。

**1. 规格覆盖**——逐节对照，全部有对应任务：

| 规格章节 | 实现任务 |
|---|---|
| 2.2 SSE 协议 / `awaiting_approval` 阻塞语义 | Task 2、5、6 |
| 2.3 审批接口（含 `run_not_found`） | Task 4、6 |
| 2.4 / 2.5 文档与知识库接口 | Task 4、10、11 |
| 2.6 ①无 CORS → 单源 | Task 12 |
| 2.6 ②无会话列表 → 前端持久化 | Task 8 |
| 2.6 ③无任务查询 → 不做批量上传 | Task 11（含测试断言 `not.toContain("upload-bulk")`） |
| 2.7 路径拼写陷阱 | Task 4（含专项测试） |
| 3.1 版本下限 / 3.2 三个陷阱 | Task 1（Tailwind v4、单一 router 入口） |
| 4.2 目录结构 | Task 1–11 的 Files 块 |
| 4.3 应用外壳 / 左栏随区切换 | Task 9 |
| 5.1 全宽文档式 + 降级差异化 | Task 7 |
| 5.2 状态机 | Task 6 |
| 5.3 会话持久化 + 刷新恢复 | Task 8、9（Task 13 Step 4 实测） |
| 5.4 待审批三层可见性 | Task 9 |
| 5.5 / 5.6 两个管理区 | Task 10、11 |
| 6 错误与边界（除下面 4 条外） | 各任务 |
| 7 测试策略 | 全部任务的 Test 文件 |

**2. 占位符扫描**——已扫，无 `TBD` / `TODO` / "稍后补充"。
唯一一处二选一是 Task 10 Step 4 的删除确认方式（`window.confirm` vs shadcn `Dialog`），
已明确要求"选定一种并让测试与之匹配，不要两套并存"。

**3. 类型一致性**——已核对：
`ChatMessage`（Task 6 定义，Task 7/8 消费）、
`ApprovalRequest`（Task 6 定义，Task 7 消费）、
`ApprovalDecision`（Task 4 定义，Task 6/7 消费）、
`PendingApproval` / `PendingApprovalsResponse`（Task 4 定义，Task 9 消费）、
`Session`（Task 8 定义，Task 9/10 消费）。
`useSessionList` / `useKbCollectionList` / `useGraphCollectionList` 在 Task 9 被引用、
在 Task 10/11 定义——**执行顺序上 Task 9 早于 Task 10/11**，
因此实现 Task 9 时若这三个 hook 尚不存在，需先建空壳再在后续任务补实现。
这是本计划唯一的顺序耦合，已在此显式说明。

**4. Review Focus 落位**——5 条全部有对应测试：

| # | 失效模式 | 测试所在任务 |
|---|---|---|
| 1 | SSE 事件跨 chunk 被切开 | Task 2（`跨 chunk 被切开的同一条事件拼回来`） |
| 2 | 审批失效 `run_not_found` | Task 6（`审批已失效…给出明确提示`） |
| 3 | 中文文件名未编码 | Task 10（`删除中文文件名时做 URL 编码`）+ Task 13 Step 5 实测 |
| 4 | dev proxy 配错 | Task 1 Step 11（人工验收 fetch 状态码） |
| 5 | 左栏轮询反复重建 | Task 9（固定 query key + 单一实例约定） |

**5. 范围**——13 个任务，每个都有独立可测的交付物。
未拆分（依第 11 节的范围结论：三个区共用外壳与数据层，拆开会产生三份重复任务）。

