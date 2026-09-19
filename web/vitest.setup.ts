// 引入 jest-dom 的断言扩展（toBeInTheDocument 等）。
// 必须在每个测试文件之前执行，由 vite.config.ts 的 test.setupFiles 挂载。
import "@testing-library/jest-dom/vitest";

// ⚠️ jsdom 不实现 scrollIntoView。MessageList 在每次内容更新时都会调用它，
//    不补桩的话任何渲染到消息列表的测试都会抛 "scrollIntoView is not a function"。
//    浏览器里它本来就有，这里只是补齐 jsdom 的缺口（不是给生产代码打补丁）。
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
