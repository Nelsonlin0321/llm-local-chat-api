import os
import boto3
import shutil
from uuid import uuid4
from fastapi import APIRouter, File,UploadFile
from app.routers.baichuan2_13b_chat_4bits.endpoints import model
from .payload import RetrievalLoad
from . import utils
from .chroma_engine import ChromaEngine


ROUTE_NAME = "RAG"
FILES_DIR = "/home/clive/workspace/llm-local-chat-api/app/files"

router = APIRouter(
    prefix=f"/{ROUTE_NAME}",
    tags=[ROUTE_NAME],
)


chroma_engine = ChromaEngine()

s3 = boto3.client('s3')


def upload_file_to_s3(file_path, object_name):
    s3.upload_file(file_path, "cloudfront-aws-bucket",
                    os.path.join("chinese-local-rag", object_name))


def save_file(file):
    file_path = f"{FILES_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return file_path


@router.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...)):
    save_file_path = save_file(file=file)
    
    upload_file_to_s3(save_file_path,os.path.basename(save_file_path))

    message = chroma_engine.ingest_file(
        file_path=save_file_path,
        sentence_size=256,
        overlapping_num=2,
        overwrite=True)
    
    return message


@router.post("/retrieval_generate")
async def retrieval_generate(pay_load: RetrievalLoad):
    # try:
    if pay_load.context:
        context = pay_load.context
    else:
        context = pay_load.question

    retrieved_result = chroma_engine.vector_search(
        pay_load.file_name, query_text=context, limit=5)

    prompt = utils.generate_prompt(retrieved_result,question=pay_load.question)

    messages = [
        {'role': 'user"',
         'content': prompt,
         }
    ]

    completion = model.generate_answer(
        model_name="Baichuan2-13B", messages=messages)

    answer = completion['choices'][0]['message']['content']

    return {
        "context": pay_load.context,
        "question": pay_load.question,
        "file_name": pay_load.file_name,
        "answer": answer,
        "uuid": str(uuid4())}


@router.get("/documents")
async def get_documents():
    documents = chroma_engine.list_document()

    doc_metas = documents['metadatas']
    count = len(doc_metas)

    return {"results":doc_metas,"count":count}
