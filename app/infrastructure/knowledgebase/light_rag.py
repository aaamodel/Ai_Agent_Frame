import os
import asyncio

from lightrag.utils import EmbeddingFunc
from pypdf import PdfReader
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from langchain_core.tools import tool

from app.config import get_settings

# ==============================================================================
# 1. 核心环境与阿里百炼参数配置
# ==============================================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

light_rag_settings = get_settings()
llm_model= light_rag_settings.llm_tier_standard or light_rag_settings.llm_tier_fast
print(f"light_rag的llm_model是:{llm_model}")
EMBEDDING_MODEL = "text-embedding-v3"

# 💡 优化：确保这些目录无论在哪个路径下启动 FastAPI 都能正确对应
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(BASE_DIR, "raw_data", "lightrag_workspace")

# 确保 RAG 工作目录存在
os.makedirs(WORKING_DIR, exist_ok=True)

# ==============================================================================
# 2. 适配百炼 Qwen 的 LightRAG 核心基础函数
# ==============================================================================
async def qwen_llm_complete(
        prompt, lightrag_system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        llm_model,
        prompt,
        system_prompt=lightrag_system_prompt,  #这里应该是传入给大模型进行实体提取的提示词，按理lightrag后台会有默认的模板
        history_messages=history_messages,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        extra_body={"enable_thinking": False},
        **kwargs
    )

async def qwen_embedding(texts: list[str]) -> list[list[float]]:
    return await openai_embed(
        texts,
        model=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )

# 1. 包装百炼的向量函数
wrapped_embedding = EmbeddingFunc(
    embedding_dim=1024,
    func=qwen_embedding,
    max_token_size=2048,
    supports_asymmetric=False
)

# 2. 初始化 LightRAG 实例（供外部全局调用）
rag_instance = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=qwen_llm_complete,
    llm_model_name=llm_model,
    embedding_func=wrapped_embedding
)

# ==============================================================================
# 3. 数据处理流：将本地的 PDF / TXT 转化为纯文本并入库
# ==============================================================================
def parse_local_file(file_path: str) -> str:
    """提取本地单份 PDF 或 TXT 文件的文本内容"""
    ext = os.path.splitext(file_path)[-1].lower()
    knowledgebase_text = ""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            knowledgebase_text = f.read()
    elif ext == ".pdf":
        pdf_reader = PdfReader(file_path)
        extracted_pages = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)
        knowledgebase_text = "\n".join(extracted_pages)

    return knowledgebase_text.strip()

async def ingest_data_pipeline(data_directory: str):
    """
    【清洗与建库】扫描目标文件夹，提取文本并批量织入 LightRAG 知识图谱
    项目初始化（冷启动）：比如你手头已经有 100 份业务文档，你可以直接把它们丢进 ./raw_data 目录，然后直接在命令行运行 python light_rag.py。它会调用 ingest_data_pipeline 帮你把基础知识库一次性建好。

    做一个“同步历史文件”的专用接口：如果你希望在网页上提供一个“同步本地文件夹”的按钮，你可以专门为它写一个新的 Router 接口。代码长这样：

    """
    if not os.path.exists(data_directory):
        os.makedirs(data_directory, exist_ok=True)
        print(f"ℹ️ 自动创建了原始数据存放目录 '{data_directory}'")
        return

    all_texts = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.lower().endswith(('.pdf', '.txt')):
                file_path = os.path.join(root, file)
                print(f"📂 正在解析本地文件: {file_path}")
                content = parse_local_file(file_path)
                if content:
                    all_texts.append(content)

    if all_texts:
        print(f"🧠 开始向 LightRAG 注入数据并提取图谱关系（共 {len(all_texts)} 份文档）...")
        await rag_instance.initialize_storages()
        await rag_instance.ainsert(all_texts)
        print("✅ 知识图谱构建完成！")
    else:
        print("⚠️ 未在目标目录下找到有效的 PDF 或 TXT 文件。")



# ==============================================================================
# 5. 统一调度执行流程（仅在直接运行此文件时生效，不影响 FastAPI）
# ==============================================================================
async def main():
    # 保持本地测试目录和全局一致
    raw_data_dir = os.path.join(BASE_DIR, "raw_data")
    await ingest_data_pipeline(raw_data_dir)

if __name__ == "__main__":

    asyncio.run(main())