# Agent 控制台前端

`Enterprise_aiagent` 的 Web 控制台。三个区：**对话**（含危险工具审批）、**文档**（RAG 集合）、**知识库**（图谱集合）。

## 开发

```bash
npm install
npm run dev     # http://localhost:5173，/api 经 vite proxy 转发到 127.0.0.1:8000
npm test        # vitest（--run 走一次性模式）
npm run build   # tsc -b && vite build → dist/
```

后端需同时运行：`uvicorn app.main:app --reload`

## 日常使用（单进程、同源、无 CORS）

```bash
npm run build                   # 产出 web/dist/
# 然后只起后端：uvicorn app.main:app
# 访问 http://127.0.0.1:8000/
```

后端会把 `web/dist` 挂到 `/`，并对**非 `/api/` 前缀**的未知路径回退 `index.html`
（前端路由如 `/documents/sales_kb` 刷新不会 404）。因此前后端同源，**不需要任何 CORS 配置**。

`web/dist` 不存在时后端只打一条 WARNING 并跳过挂载，**照常启动**——前端没构建也能起服务。

## 四个容易踩的坑

1. **Tailwind 是 v4：没有 `tailwind.config.js`。**
   主题写在 `src/index.css` 的 `@theme {}` 里，插件用 `@plugin` 挂载。
   按 v3 写法建配置文件会**静默失效**（不报错，样式就是不生效）。

2. **图谱单文件上传的路径是 `kownledgebase`（少一个 w）。**
   这是后端 `app/api/routes/kownledgebase.py` 的实际拼写，**不要"顺手改成正确拼写"**，改了就是 404。
   隔壁 `/documents/knowledgebase/upload-bulk` 才是正确拼写，但那是批量接口
   （返回 202 后台任务，且后端没有任务进度查询端点），本期不使用。

3. **SSE 不能用 `EventSource`。**
   `/chat/with_agent` 是 POST，而 `EventSource` 只支持 GET。
   解析在 `src/lib/sse.ts`，要自己处理"一条事件被 TCP 从中间切开"的情况。

4. **审批是循环，不是一次性的。**
   命中危险工具时后端推 `awaiting_approval` 并**结束本轮流**（此时对话没有答案）。
   前端必须调 `POST /agent/runs/{run_id}/approval` 续流，而**续跑中可能再次挂起**同一 `run_id`。
   不处理就是对话永久卡死。

## 会话历史

存在浏览器 localStorage（`agent-console.sessions`）。**后端没有会话列表接口**，
它只持有短期记忆——"我上次问了什么"只能由前端自己记。

会话由 URL 的 `?session=<id>` 标识；不带参数时自动建新会话并 `replace` 写回 URL。

## 待办 / 已知限制

- 三区目前都是单页实现，未做代码分割（构建产物单 chunk 约 527 kB / gzip 162 kB）。
- 无鉴权（后端也没有认证中间件），仅适合内网/本机使用。
- 知识库区不支持批量上传，原因见上面第 2 条。
