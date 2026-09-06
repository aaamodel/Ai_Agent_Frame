# -*- coding: utf-8 -*-
import json
from typing import Any
import httpx
from app.core.tools.base import BaseTool, ToolParameter


class FeishuBitableTool(BaseTool):
    """读写飞书多维表格的工具"""

    name: str = "feishu_bitable_tool"
    description: str = "用于从飞书多维表格（Bitable）中读取记录或新增数据行。"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = [
            ToolParameter(name="app_id", type="string", description="飞书开放平台应用的 App ID", required=True),
            ToolParameter(name="app_secret", type="string", description="飞书开放平台应用的 App Secret", required=True),
            ToolParameter(name="app_token", type="string",
                          description="多维表格的 app_token (从多维表格的浏览器 URL 中截取)", required=True),
            ToolParameter(name="table_id", type="string", description="具体数据表的 table_id (通常以 tbl 开头)",
                          required=True),
            ToolParameter(name="action", type="string",
                          description="操作类型: 'list' (查询多条数据) 或 'add' (新增一行记录)", required=True),
            ToolParameter(name="fields_json", type="string",
                          description="新增记录时的 JSON 字符串键值对 (例如: '{\"姓名\": \"张三\", \"金额\": 200}')",
                          required=False)
        ]

    async def _fetch_tenant_token(self, app_id: str, app_secret: str) -> str:
        """鉴权方法：获取飞书临时凭证"""
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={"app_id": app_id, "app_secret": app_secret})
            resp.raise_for_status()
            return resp.json().get("tenant_access_token", "")

    async def execute(self, **kwargs: Any) -> str:
        app_id = kwargs.get("app_id")
        app_secret = kwargs.get("app_secret")
        app_token = kwargs.get("app_token")
        table_id = kwargs.get("table_id")
        action = kwargs.get("action")
        fields_json = kwargs.get("fields_json")

        try:
            # 1. 动态鉴权
            token = await self._fetch_tenant_token(app_id, app_secret)
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
            api_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"

            async with httpx.AsyncClient() as client:
                # 2. 执行列出数据
                if action == "list":
                    resp = await client.get(api_url, headers=headers)
                    res_data = resp.json()
                    if res_data.get("code") != 0:
                        return f"飞书接口调用失败: {res_data.get('msg')}"

                    items = res_data.get("data", {}).get("items", [])
                    # 简化抽取关键内容，防止大模型上下文爆掉
                    cleaned_records = [{"record_id": i["record_id"], "fields": i["fields"]} for i in items]
                    return json.dumps(cleaned_records, ensure_ascii=False, indent=2)

                # 3. 执行写入数据
                elif action == "add":
                    if not fields_json:
                        return "错误：使用 add 动作时必须提供 fields_json 数据参数"

                    fields = json.loads(fields_json)
                    payload = {"fields": fields}
                    resp = await client.post(api_url, json=payload, headers=headers)
                    res_data = resp.json()
                    if res_data.get("code") != 0:
                        return f"飞书写入失败: {res_data.get('msg')}"

                    return f"成功：数据已写入飞书多维表，生成的 Record_ID 为: {res_data['data']['record']['record_id']}"

                else:
                    return f"错误：未知的操作类型 '{action}'"
        except Exception as e:
            return f"飞书多维表工具执行异常: {str(e)}"