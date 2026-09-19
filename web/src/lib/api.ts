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
