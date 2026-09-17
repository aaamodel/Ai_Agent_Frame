# ============================================================================
# intent_tree.py - 意图树缓存与工厂构建
#
# 设计定位（Agent 编排语义）：
#   意图树不再是"RAG 知识库路由表"（kb_id/collection_name 导向），而是
#   "Agent 工具路由表"——叶子节点 = 一类用户需求场景，节点上的
#   agent_tool_names = 该场景下"大概率能拿到结果"的注册工具
#   （与 rag_constant.REGISTERED_ENABLED_TOOL_NAMES 1:1 对齐）。
#
#   三通道语义重定义（IntentKind 枚举 code 不变，兼容 DB 旧数据）：
#     KB     = 内部知识检索类意图（rag_knowledge_search / knowledge_graph_search）
#     MCP    = 操作型工具意图（web_search / excel / bitable / file_*）
#     SYSTEM = 无工具闲聊意图（welcome / about_bot）
#
#   树结构对标旧 RAG 领域树（DOMAIN → CATEGORY/TOPIC → 叶子），保留业务领域
#   划分（人事/IT/财务/业务系统/销售数据），但每个叶子映射到真实注册工具。
# ============================================================================
from __future__ import annotations

import json
import logging
from typing import Any

from app.query_intent.intent_classify_resolver.intent_model import IntentNode
from app.query_intent.intent_data_base import IntentKind, IntentLevel

logger = logging.getLogger(__name__)


class IntentTreeCacheManager:
    INTENT_TREE_CACHE_KEY: str = "ragent:intent_classify_resolver:tree"
    CACHE_EXPIRE_DAYS: int = 7

    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis_client = redis_client
        self._local_cache: dict[str, list[IntentNode]] = {}

    def get_intent_tree_from_cache(self) -> list[IntentNode] | None:
        try:
            if self._redis_client is not None:
                cache_json = self._redis_client.get(IntentTreeCacheManager.INTENT_TREE_CACHE_KEY)
            else:
                cache_json = self._local_cache.get(IntentTreeCacheManager.INTENT_TREE_CACHE_KEY)
                if cache_json is not None:
                    cache_json = json.dumps([self._node_to_dict(n) for n in cache_json])

            if cache_json is None:
                logger.info("意图树缓存不存在，需要从数据库加载")
                return None

            data = json.loads(cache_json)
            return [self._dict_to_node(d) for d in data]
        except Exception as e:
            logger.error("从Redis读取意图树缓存失败", exc_info=e)
            return None

    def save_intent_tree_to_cache(self, roots: list[IntentNode]) -> None:
        try:
            cache_json = json.dumps([self._node_to_dict(n) for n in roots], ensure_ascii=False)
            if self._redis_client is not None:
                self._redis_client.set(
                    IntentTreeCacheManager.INTENT_TREE_CACHE_KEY,
                    cache_json,
                    ex=IntentTreeCacheManager.CACHE_EXPIRE_DAYS * 86400,
                )
            else:
                self._local_cache[IntentTreeCacheManager.INTENT_TREE_CACHE_KEY] = roots
            logger.info("意图树已保存到Redis缓存，根节点数: %d", len(roots))
        except Exception as e:
            logger.error("保存意图树到Redis缓存失败", exc_info=e)

    def clear_intent_tree_cache(self) -> None:
        deleted = False
        if self._redis_client is not None:
            deleted = bool(self._redis_client.delete(IntentTreeCacheManager.INTENT_TREE_CACHE_KEY))
        else:
            if IntentTreeCacheManager.INTENT_TREE_CACHE_KEY in self._local_cache:
                del self._local_cache[IntentTreeCacheManager.INTENT_TREE_CACHE_KEY]
                deleted = True
        if deleted:
            logger.info("意图树缓存已清除，Key: %s", IntentTreeCacheManager.INTENT_TREE_CACHE_KEY)
        else:
            logger.info("意图树缓存不存在，无需清除")

    def is_cache_exists(self) -> bool:
        try:
            if self._redis_client is not None:
                return bool(self._redis_client.exists(IntentTreeCacheManager.INTENT_TREE_CACHE_KEY))
            else:
                return IntentTreeCacheManager.INTENT_TREE_CACHE_KEY in self._local_cache
        except Exception as e:
            logger.error("检查意图树缓存是否存在失败", exc_info=e)
            return False

    @staticmethod
    def _node_to_dict(node: IntentNode) -> dict:
        return {
            "id": node.id,
            "kb_id": node.kb_id,
            "name": node.name,
            "description": node.description,
            "level": node.level.value if node.level else None,
            "parent_id": node.parent_id,
            "examples": node.examples,
            "children": [IntentTreeCacheManager._node_to_dict(c) for c in (node.children or [])],
            "full_path": node.full_path,
            "kind": node.kind.value if node.kind else None,
            "collection_name": node.collection_name,
            "collection_names": node.collection_names,
            "mcp_tool_id": node.mcp_tool_id,
            "top_k": node.top_k,
            "prompt_snippet": node.prompt_snippet,
            "prompt_template": node.prompt_template,
            "param_prompt_template": node.param_prompt_template,
            # ---- Agent 编排工具路由字段（新增；旧缓存缺字段时 .get 兜底为空） ----
            "agent_tool_names": node.agent_tool_names,
            "tool_usage_hint": node.tool_usage_hint,
            "prefer_mode": node.prefer_mode,
        }

    @staticmethod
    def _dict_to_node(data: dict) -> IntentNode:
        level = IntentLevel.from_code(data.get("level")) if data.get("level") is not None else None
        kind = IntentKind.from_code(data.get("kind")) if data.get("kind") is not None else IntentKind.KB

        node = IntentNode(
            id=data.get("id"),
            kb_id=data.get("kb_id"),
            name=data.get("name"),
            description=data.get("description"),
            level=level,
            parent_id=data.get("parent_id"),
            examples=data.get("examples", []),
            children=[],
            embedding=None,
            full_path=data.get("full_path", ""),
            kind=kind,
            collection_name=data.get("collection_name"),
            collection_names=data.get("collection_names", []),
            mcp_tool_id=data.get("mcp_tool_id"),
            top_k=data.get("top_k"),
            prompt_snippet=data.get("prompt_snippet"),
            prompt_template=data.get("prompt_template"),
            param_prompt_template=data.get("param_prompt_template"),
            agent_tool_names=data.get("agent_tool_names", []),
            tool_usage_hint=data.get("tool_usage_hint"),
            prefer_mode=data.get("prefer_mode"),
        )
        node.children = [IntentTreeCacheManager._dict_to_node(c) for c in data.get("children", [])]
        return node


class IntentTreeFactory:
    """Agent 编排领域树工厂（对标旧 RAG 领域树重写）。

    树总览（叶子节点全部挂 agent_tool_names，SYSTEM 叶子除外）：

        knowledge 企业知识问答 [KB]
        ├─ knowledge-hr              → [rag_knowledge_search]
        ├─ knowledge-it              → [rag_knowledge_search]
        ├─ knowledge-finance         → [rag_knowledge_search]
        ├─ knowledge-biz-system      → [rag_knowledge_search]
        └─ knowledge-entity-relation → [knowledge_graph_search]
        web 外部公开信息检索 [MCP]
        └─ web-live-info             → [web_search]
        data 业务数据操作 [MCP]
        ├─ data-sales-report         → [local_excel_read_tool, local_excel_query_tool, feishu_bitable_tool, sales_report_export_tool]
        ├─ data-excel-ops            → [local_excel_read_tool, local_excel_query_tool, local_excel_write_tool, sales_report_export_tool]
        └─ data-bitable-ops          → [feishu_bitable_tool]
        files 本地文件操作 [MCP]
        ├─ files-locate              → [file_list_tool]
        └─ files-content             → [file_grep_tool, file_read_tool]
        sys 系统交互 [SYSTEM]
        ├─ sys-welcome / sys-about-bot（无工具）
    """

    # ---- 内部知识检索（对标旧 group/biz 域：答案在公司私有文档中检索得到） ----
    _HINT_RAG_SEARCH: str = (
        "调用 rag_knowledge_search，query 传改写后的完整问题；"
        "答案来自公司私有文档知识库，检索不到时如实告知，不要编造。"
    )
    _HINT_GRAPH_SEARCH: str = (
        "调用 knowledge_graph_search，query 传实体+关系描述"
        "（如 'OA系统 与 保险系统 的关系'）；适合'谁负责/什么关系/上下游'类问题。"
    )
    _HINT_WEB_SEARCH: str = (
        "调用 web_search，query 传检索意图，可用 count(1-10) 与 time_range"
        "(OneDay/OneWeek/OneMonth/OneYear) 限定条数与时效；"
        "该工具是**唯一**的联网检索工具，内部自带重试与降级，无需（也无法）切换到其他搜索工具。"
    )
    _HINT_EXCEL: str = (
        "精确取数/统计/排名/占比/环比等'要算答案'的问题优先用 local_excel_query_tool："
        "传 file_path + 一句中文 query（多 sheet 给 sheet_name），它在**全量数据**上跑 pandas，"
        "目标数据不在前几行绝不代表不存在，不要因预览误判数据缺失。"
        "只看表结构/列名用 local_excel_read_tool（不传 sheet_name 即返回全部 sheet 的"
        "列名/行列数/前几行摘要）；看某条记录用 filter_column+filter_value（多 sheet 须配 sheet_name）。"
        "若不知道文件路径，先用 file_list_tool（默认递归 3 层，可按 pattern 过滤）定位文件。"
        "写入用 local_excel_write_tool：改某条记录传 filter_column+filter_value+target_column+new_value"
        "（坐标工具内部算，禁止自己数 A1 坐标）；一次写多行传 rows（JSON 数组）。"
        "写操作会触发人工审批，写入前必须先读定位并向用户确认旧值→新值。"
    )
    _HINT_SALES_REPORT_EXPORT: str = (
        "用户明确要求导出/生成/下载/归档销售报表或分析报告文件时，"
        "调用 sales_report_export_tool 把分析结论生成 Excel 落到 outputs/sales_reports/；"
        "该工具是写操作，会触发人工审批，普通问答不要调用。"
    )
    _HINT_BITABLE: str = (
        "调用 feishu_bitable_tool 操作飞书多维表格；"
        "需要 app_token 与 table_id（可从用户问题或上下文获取），"
        "缺关键参数时先向用户确认，不要猜测。"
    )
    _HINT_FILE_LIST: str = (
        "调用 file_list_tool 浏览目录/定位文件：默认一次递归 3 层（depth 可调，最大 8），"
        "可用 pattern（如 '*.xlsx'）过滤文件名——**不要一层层反复调用去摸路径**；"
        "找到目标文件后通常接 file_read_tool / local_excel_read_tool 深入内容。"
    )
    _HINT_FILE_CONTENT: str = (
        "先调用 file_grep_tool 按关键词/正则在文件内容中检索定位，"
        "再调用 file_read_tool 读取命中文件的具体片段。"
    )
    @staticmethod
    def build_intent_tree() -> list[IntentNode]:
        roots: list[IntentNode] = []

        # ================================================================
        # 1. 企业知识问答（KB 通道：内部知识检索类工具大概率能拿到结果）
        #    对标旧 RAG 领域树的"集团信息化 / 业务系统"业务划分。
        # ================================================================
        knowledge = IntentNode(
            id="knowledge",
            name="企业知识问答",
            level=IntentLevel.DOMAIN,
            kind=IntentKind.KB,
            description="答案存在于公司内部文档/知识库/知识图谱中的问题",
        )

        knowledge_hr = IntentNode(
            id="knowledge-hr",
            name="人事制度",
            level=IntentLevel.CATEGORY,
            parent_id=knowledge.id,
            kind=IntentKind.KB,
            description="招聘、入职、转正、离职、绩效、薪资、考勤、请假、报销差旅等公司制度与流程问题；答案在公司制度文档中大概率检索得到",
            examples=["请假流程是怎样的？", "试用期多久转正？", "差旅报销标准是多少？"],
            agent_tool_names=["rag_knowledge_search"],
            tool_usage_hint=IntentTreeFactory._HINT_RAG_SEARCH,
        )

        knowledge_it = IntentNode(
            id="knowledge-it",
            name="IT支持",
            level=IntentLevel.CATEGORY,
            parent_id=knowledge.id,
            kind=IntentKind.KB,
            description="VPN、邮箱、打印机、网络、电脑账号密码、办公软件使用等 IT 支持类问题；答案在 IT 知识库文档中大概率检索得到",
            examples=["电脑打印机怎么连？", "公司 VPN 连不上怎么办？", "邮箱密码忘了怎么重置？"],
            agent_tool_names=["rag_knowledge_search"],
            tool_usage_hint=IntentTreeFactory._HINT_RAG_SEARCH,
        )

        knowledge_finance = IntentNode(
            id="knowledge-finance",
            name="财务发票",
            level=IntentLevel.CATEGORY,
            parent_id=knowledge.id,
            kind=IntentKind.KB,
            description="报销、付款、成本中心、预算、发票抬头、纳税资质、纳税人识别号等财务制度与发票信息问题；答案在财务文档中大概率检索得到",
            examples=["公司的发票抬头有哪些？", "纳税人识别号是多少？", "成本中心怎么申请？"],
            agent_tool_names=["rag_knowledge_search"],
            tool_usage_hint=IntentTreeFactory._HINT_RAG_SEARCH,
        )

        knowledge_biz_system = IntentNode(
            id="knowledge-biz-system",
            name="业务系统文档",
            level=IntentLevel.CATEGORY,
            parent_id=knowledge.id,
            kind=IntentKind.KB,
            description="OA 系统、保险系统等内部业务系统的功能介绍、架构设计、数据权限与安全说明类问题；答案在系统文档中大概率检索得到",
            examples=["OA系统主要提供哪些功能？", "保险系统整体架构是怎样的？", "OA系统的权限如何控制？"],
            agent_tool_names=["rag_knowledge_search"],
            tool_usage_hint=IntentTreeFactory._HINT_RAG_SEARCH,
        )

        knowledge_entity_relation = IntentNode(
            id="knowledge-entity-relation",
            name="实体关系图谱",
            level=IntentLevel.CATEGORY,
            parent_id=knowledge.id,
            kind=IntentKind.KB,
            description="实体之间关系类问题：系统/部门/人员/项目之间的负责关系、上下游关系、协作关系（A 和 B 是什么关系、谁负责 X、X 的上游是什么）；答案在知识图谱中大概率检索得到，普通文档检索对这类问题效果差",
            examples=["OA系统和保险系统是怎么对接的？", "销售部负责人是谁？", "这个项目的上游依赖是什么？"],
            agent_tool_names=["knowledge_graph_search"],
            tool_usage_hint=IntentTreeFactory._HINT_GRAPH_SEARCH,
        )

        # ⚠️ 这个叶子是补缺口用的：公司自有语料（rag_data/ 13 篇销售语料）主体是
        # 《公司产品知识库》《产品与报价表》，但在它存在之前，KB 分支只有
        # 人事/IT/财务/业务系统/图谱五个叶子——任何「产品·定价·折扣·部署交付·服务承诺」
        # 类问题都只能被塞进最近邻（实测：「客户要私有化部署，一般多久能上线？实施费怎么收？」
        # 被判成「财务发票」，「准确率承诺写不写进合同」被判成「业务系统文档」）。
        # 这是**结构性缺节点**，不是模型能力问题。
        knowledge_product = IntentNode(
            id="knowledge-product",
            name="产品与报价",
            level=IntentLevel.CATEGORY,
            parent_id=knowledge.id,
            kind=IntentKind.KB,
            description="公司自有产品线（智能客服平台/工单系统/企业知识库）的版本与定价、折扣与报价权限、组合销售与打包优惠、部署与交付（SaaS 与私有化上线周期、实施服务费）、服务承诺（准确率与 SLA 是否写进合同）、产品 FAQ 与竞品差异化话术等销售产品类问题；答案在公司产品知识库中大概率检索得到",
            examples=["标准版和专业版有什么区别？", "折扣权限是怎么规定的？", "实施服务费是怎么收的？"],
            agent_tool_names=["rag_knowledge_search"],
            tool_usage_hint=IntentTreeFactory._HINT_RAG_SEARCH,
        )

        knowledge.children = [
            knowledge_hr,
            knowledge_it,
            knowledge_finance,
            knowledge_biz_system,
            knowledge_entity_relation,
            knowledge_product,
        ]
        roots.append(knowledge)

        # ================================================================
        # 2. 外部公开信息检索（MCP 通道：联网搜索大概率能拿到结果）
        # ================================================================
        web = IntentNode(
            id="web",
            name="外部公开信息检索",
            level=IntentLevel.DOMAIN,
            kind=IntentKind.MCP,
            description="需要联网检索外部公开信息才能回答的问题",
        )

        web_live_info = IntentNode(
            id="web-live-info",
            name="实时公开信息",
            level=IntentLevel.CATEGORY,
            parent_id=web.id,
            kind=IntentKind.MCP,
            description="新闻资讯、天气、汇率股价、行业动态、政策法规、公开公司信息、技术资料等外部实时/公开信息问题；内部知识库没有这些数据，联网搜索是唯一大概率拿到结果的途径",
            examples=["今天天气怎么样？", "现在美元汇率是多少？", "最近 AI 行业有什么大新闻？"],
            agent_tool_names=["web_search"],
            tool_usage_hint=IntentTreeFactory._HINT_WEB_SEARCH,
        )

        web.children = [web_live_info]
        roots.append(web)

        # ================================================================
        # 3. 业务数据操作（MCP 通道：表格/多维表格工具大概率能拿到结果）
        #    对标旧 sales（销售汇总数据统计）MCP 域——数据载体统一映射到
        #    真实注册的 Excel / 飞书多维表格工具。
        # ================================================================
        data = IntentNode(
            id="data",
            name="业务数据操作",
            level=IntentLevel.DOMAIN,
            kind=IntentKind.MCP,
            description="需要对表格/多维表格中的业务数据进行查询、统计、写入的问题",
        )

        data_sales_report = IntentNode(
            id="data-sales-report",
            name="销售数据统计",
            level=IntentLevel.CATEGORY,
            parent_id=data.id,
            kind=IntentKind.MCP,
            description="销售总额、销售量、销售占比、销售趋势、排行榜等业务数据统计问题；这类数据存在表格/多维表格中，用表格工具读取统计大概率能拿到结果；明确要求导出/生成报表文件时用 sales_report_export_tool",
            examples=["这个月的销售总额是多少？", "各区域销量占比怎么样？", "上季度销量 Top10 有哪些？", "把本月销售分析导出成报表"],
            agent_tool_names=["local_excel_read_tool", "feishu_bitable_tool", "sales_report_export_tool"],
            tool_usage_hint=IntentTreeFactory._HINT_EXCEL + IntentTreeFactory._HINT_SALES_REPORT_EXPORT,
        )

        data_excel_ops = IntentNode(
            id="data-excel-ops",
            name="本地表格处理",
            level=IntentLevel.CATEGORY,
            parent_id=data.id,
            kind=IntentKind.MCP,
            description="对本地 Excel 文件的读取、筛选、汇总、透视、写入、格式处理等操作类请求",
            examples=["帮我读一下 sales.xlsx 里的数据", "把这张表按月份汇总", "在表格里新增一行记录", "把分析结果导出成 Excel 报表"],
            agent_tool_names=["local_excel_read_tool", "local_excel_query_tool", "local_excel_write_tool", "sales_report_export_tool"],
            tool_usage_hint=IntentTreeFactory._HINT_EXCEL + IntentTreeFactory._HINT_SALES_REPORT_EXPORT,
        )

        data_bitable_ops = IntentNode(
            id="data-bitable-ops",
            name="多维表格操作",
            level=IntentLevel.CATEGORY,
            parent_id=data.id,
            kind=IntentKind.MCP,
            description="对飞书多维表格（Bitable）记录的查询、新增、更新、删除等操作类请求",
            examples=["帮我在多维表格里加一条记录", "查一下多维表格里本月的待办", "更新多维表格里这条数据的状态"],
            agent_tool_names=["feishu_bitable_tool"],
            tool_usage_hint=IntentTreeFactory._HINT_BITABLE,
        )

        data.children = [data_sales_report, data_excel_ops, data_bitable_ops]
        roots.append(data)

        # ================================================================
        # 4. 本地文件操作（MCP 通道：文件工具大概率能拿到结果）
        # ================================================================
        files = IntentNode(
            id="files",
            name="本地文件操作",
            level=IntentLevel.DOMAIN,
            kind=IntentKind.MCP,
            description="对本地工作区文件的浏览、定位、内容检索、读取类请求",
        )

        files_locate = IntentNode(
            id="files-locate",
            name="文件定位浏览",
            level=IntentLevel.CATEGORY,
            parent_id=files.id,
            kind=IntentKind.MCP,
            description="查看目录结构、列出文件、找某个文件在哪里、看文件名/类型/大小等文件定位类请求",
            examples=["项目根目录下有哪些文件？", "帮我找一下配置文件在哪", "列出 docs 目录的内容"],
            agent_tool_names=["file_list_tool"],
            tool_usage_hint=IntentTreeFactory._HINT_FILE_LIST,
        )

        files_content = IntentNode(
            id="files-content",
            name="文件内容检索",
            level=IntentLevel.CATEGORY,
            parent_id=files.id,
            kind=IntentKind.MCP,
            description="在代码/文本文件内容中按关键词搜索、查看某个文件的具体内容等文件内容类请求",
            examples=["帮我在代码里搜一下 build_intent_tree 的定义", "读一下 requirements.txt 的内容", "哪个文件里用到了 ToolCallBudget？"],
            agent_tool_names=["file_grep_tool", "file_read_tool"],
            tool_usage_hint=IntentTreeFactory._HINT_FILE_CONTENT,
        )

        files.children = [files_locate, files_content]
        roots.append(files)

        # ================================================================
        # 5. 任务规划（MCP 通道）—— 已整体移除（2026-09-16）
        #
        # 原 `task` / `task-todo-plan` 叶子的唯一工具是 write_todos，该工具已从架构中
        # 删除。它的原型是 deepagents 的 TodoListMiddleware，成立前提有三条，本项目
        # 全不具备：(a) 暴露 todos state channel；(b) 每轮把当前清单回注上下文；
        # (c) 面向长 horizon 循环。本项目里它只是一次无状态、无人消费的 observation，
        # 且会被 `compact_tool_observations` 在 3 轮后压成短桩。
        # 多步骤规划由 plan_execute 的 Planner 子任务账本承担，无需本通道。
        # ================================================================

        # ================================================================
        # 6. 系统交互（SYSTEM 通道：无工具，闲聊/问候）
        # ================================================================
        sys = IntentNode(
            id="sys",
            name="系统交互",
            level=IntentLevel.DOMAIN,
            kind=IntentKind.SYSTEM,
        )

        welcome = IntentNode(
            id="sys-welcome",
            name="欢迎与问候",
            level=IntentLevel.CATEGORY,
            parent_id=sys.id,
            description="用户与助手打招呼，如：你好、早上好、hi、在吗 等",
            examples=["你好", "hello", "早上好", "在吗", "嗨"],
            kind=IntentKind.SYSTEM,
        )

        about_bot = IntentNode(
            id="sys-about-bot",
            name="关于助手",
            level=IntentLevel.CATEGORY,
            parent_id=sys.id,
            description="询问助手是做什么的、是谁、能做什么等",
            examples=["你是谁", "你是做什么的", "你能帮我做什么", "你是什么AI"],
            kind=IntentKind.SYSTEM,
        )

        sys.children = [welcome, about_bot]
        roots.append(sys)

        IntentTreeFactory._fill_full_path(roots, None)
        return roots

    @staticmethod
    def _fill_full_path(nodes: list[IntentNode], parent: IntentNode | None) -> None:
        for node in nodes:
            if parent is None:
                node.full_path = node.name or ""
            else:
                node.full_path = (parent.full_path or "") + " > " + (node.name or "")
            if node.children is not None and len(node.children) > 0:
                IntentTreeFactory._fill_full_path(node.children, node)
