import { describe, expect, it } from "vitest";

import * as approvals from "./approvals";
import * as docs from "./documents";
import * as kb from "./knowledgebase";

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
