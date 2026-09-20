import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Agent 回答常含表格 / 代码块 / 长路径，统一走 Markdown 渲染。
 *
 * ⚠️ `prose-invert` 是深色底必需的：默认的 prose 颜色是深灰字，
 *    在深色面板上会几乎看不见。
 */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-invert prose-sm max-w-none break-words">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}
