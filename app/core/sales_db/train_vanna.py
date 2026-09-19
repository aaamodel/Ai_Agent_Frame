# -*- coding: utf-8 -*-
"""把**口径**训练进 Vanna 的 documentation 训练数据。

用法：

              # 训练（幂等，可直接重跑）
命令：python -m app.core.sales_db.train_vanna --train-docs（幂等）
⚠️ 本模块**只管 Vanna 的训练数据**，与 RAG 知识库无关。
    规则文档（阶段流转规则 / 折扣权限 / 组合策略）要进的是项目的 Milvus 知识库
    `sales_kb`，走 `python -m app.core.sales_db.export_rule_docs` 导出后由人工上传。
    两者目的地不同、消费者不同，**不要**混在一个入口里——这个模块曾经把两者捆在
    一起（`--train-docs` / `--write-rules`），既掩盖了它们不是一回事，也让
    "Vanna 的 documentation" 和 "RAG 的文档" 这两个同义词互相污染。

为什么口径要单独喂，而不是让它走 RAG 检索（design D7b）：

    以"ICP 达标的线索有多少"为例，模型必须知道 ``ICP 达标 = 员工规模 ≥ 200`` 才能写出
    ``WHERE 员工规模 >= 200``。这个知识只在字段字典的"说明"列里，DDL 里没有。

    若走"模型先 rag_knowledge_search 再 sales_sql_query"，就等于**指望模型自觉去查**——
    这与 Excel 时代指望模型自己猜对列名是同一个错误（当时它编了 `pricing_guide`）。
    因此口径预置进 SQL 生成上下文，零额外调用、不依赖模型自觉。

实测效果：训练后问"有多少条线索的员工规模达到了ICP标准"，生成的 SQL 为
``SELECT COUNT(*) FROM 线索 WHERE 员工规模 >= 200;`` —— 那个 ``200`` 只存在于这里喂进去
的口径中，DDL 里没有。

DDL 训练**不需要**手动：见 `sql_vanna.SalesVanna.sync_ddl()`，工具首次使用时自动增量同步。
"""

from __future__ import annotations

from loguru import logger

from app.core.sales_db.knowledge import DEFAULT_GLOSSARY_HEADER, build_glossary


def train_documentation(vanna=None) -> int:
    """把口径训练进 Vanna；返回写入的条目数。

    幂等：先移除既有的 documentation 训练数据再重新训练，避免反复执行产生重复条目。
    """
    from app.core.tools.builtin.sql_vanna import get_vanna  # noqa: PLC0415 - 避免循环导入

    instance = vanna or get_vanna()

    existing = instance._vn.get_training_data()  # noqa: SLF001 - 内部 API，此处为唯一消费者
    if existing is not None and not getattr(existing, "empty", True):
        stale = existing[existing["training_data_type"] == "documentation"]
        for _, row in stale.iterrows():
            instance._vn.remove_training_data(id=row.get("id"))  # noqa: SLF001
        if not stale.empty:
            logger.info("先移除既有 documentation 训练数据 {} 条", len(stale))

    glossary = build_glossary()
    if not glossary:
        return 0
    text = DEFAULT_GLOSSARY_HEADER + "\n" + "\n".join(glossary)
    instance._vn.train(documentation=text)  # noqa: SLF001
    logger.info("已训练口径 documentation：{} 条", len(glossary))
    return len(glossary)


def _main() -> None:
    count = train_documentation()
    print(f"口径已训练进 Vanna：{count} 条")


if __name__ == "__main__":
    _main()
