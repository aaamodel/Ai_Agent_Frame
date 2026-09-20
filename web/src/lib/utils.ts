import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * 合并 Tailwind 类名：clsx 处理条件类，twMerge 消解冲突类
 * （例如 `px-2 px-4` 只留 `px-4`）。
 *
 * shadcn 的 `components.json` 把 utils 指向本文件，因此这里必须存在 ——
 * 否则以后 `npx shadcn add <组件>` 生成的代码会因找不到 `cn` 而编译失败。
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
