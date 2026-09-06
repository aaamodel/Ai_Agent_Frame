import os
import shutil
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

# LightRAG 实例与解析函数改为各路由函数内懒加载：
# 模块级 `from app.infrastructure.knowledgebase.light_rag import ...` 会在
# 应用导入期拉起 lightrag → torch / transformers 等重型依赖，拖慢启动。
def _load_light_rag():
    """懒加载 LightRAG 全局实例与文件解析函数（仅在实际调用知识库路由时触发）。"""
    from app.infrastructure.knowledgebase.light_rag import rag_instance, parse_local_file

    return rag_instance, parse_local_file



router = APIRouter(tags=["kownledgebase"])

# 定义响应模型
class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    message: str


# 设定上传文件的临时/持久存放目录
UPLOAD_DIR = "./raw_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/documents/kownledgebase/upload", response_model=DocumentUploadResponse)
async def upload_kownledgebase_document(file: UploadFile = File(...)):
    """
    上传 PDF 或 TXT 文件，解析并织入 LightRAG 本地知识库
    """
    # 1. 严格校验文件后缀
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".txt", ".pdf"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不支持的文件格式。目前仅支持 .txt 和 .pdf 文件。"
        )

    # 2. 保存上传的文件到本地目录
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            # 异步读取并写入，防止大文件阻塞主线程
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存文件到本地失败: {str(e)}"
        )
    finally:
        await file.close()  # 释放文件资源

    # 3. 调用 light_rag 的解析与入库流程
    try:
        # 提取文本内容
        content = parse_local_file(file_path)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件内容为空或无法提取有效文本。"
            )

        # 织入 LightRAG 知识图谱并固化到本地
        await rag_instance.initialize_storages()
        await rag_instance.ainsert([content])  # ainsert 接收一个包含文本的列表

        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            message="文件已成功上传，并完成 LightRAG 知识图谱的解析与本地固化！"
        )

    except Exception as e:
        # 如果解析/建库失败，自动清理掉刚才上传的破坏性或无效文件
        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LightRAG 解析或写入索引失败: {str(e)}"
        )






UPLOAD_DIR = "./raw_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 响应模型：立刻告诉前端任务已受理
class TaskResponse(BaseModel):
    status: str
    task_id: str
    message: str


async def bg_processing_pipeline(file_paths: List[str]):
    """后台默默执行的纯文本解析与 LightRAG 织入任务"""
    rag_instance, parse_local_file = _load_light_rag()
    try:
        await rag_instance.initialize_storages()

        # 批量提取所有上传文件的文本
        valid_contents = []
        for path in file_paths:
            if os.path.exists(path):
                content = parse_local_file(path)
                if content:
                    valid_contents.append(content)

        if valid_contents:
            # 批量织入 LightRAG，效率远高于一个一个 insert
            await rag_instance.ainsert(valid_contents)
            print(f"🎉 [异步任务] 成功批量织入 {len(valid_contents)} 份文档到知识库。")

            # 可以在此处将任务状态更新到数据库（如 MySQL/Redis），标记为 "success"

    except Exception as e:
        print(f"❌ [异步任务失败] 原因: {str(e)}")
        # 可以在此处将任务状态更新为 "failed"，并记录错误日志
    finally:
        # 解析完成后，清理上传的原始文件（根据业务决定是否保留）
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)


@router.post("/documents/knowledgebase/upload-bulk", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_multiple_documents(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...)
):
    """
    生产级多文件批量上传接口（兼容单文件）。
    采用异步机制，直接返回 202 状态码防浏览器超时。
    """
    if not files or files[0].filename == "":
        raise HTTPException(status_code=400, detail="未检测到上传的文件。")

    saved_file_paths = []

    # 1. 快速校验文件格式并保存到本地临时目录
    for file in files:
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".txt", ".pdf"]:
            raise HTTPException(
                status_code=400,
                detail=f"文件 {file.filename} 格式不支持。仅支持 .txt 和 .pdf"
            )

        # 💡 生产环境小细节：文件名加 UUID 唯一标识，防止多人上传同名文件导致覆盖冲突
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_file_paths.append(file_path)
        except Exception as e:
            # 如果中间某一个文件存失败了，把之前存的都删掉
            for path in saved_file_paths:
                if os.path.exists(path): os.remove(path)
            raise HTTPException(status_code=500, detail=f"保存文件失败: {str(e)}")
        finally:
            await file.close()

    # 2. 生成一个任务 ID（可以存入 Redis 用于前端轮询任务状态）
    task_id = uuid.uuid4().hex

    # 3. 将耗时的 RAG 解析任务丢进 FastAPI 后台线程池，立刻释放当前 HTTP 请求
    background_tasks.add_task(bg_processing_pipeline, saved_file_paths)

    return TaskResponse(
        status="processing",
        task_id=task_id,
        message=f"已成功接收 {len(files)} 个文件，后端正在异步解析并构建图谱，请稍后查看。"
    )