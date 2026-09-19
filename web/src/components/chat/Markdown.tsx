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
