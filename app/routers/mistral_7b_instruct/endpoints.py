from fastapi import APIRouter
from app.payload import ChatPayload
from .model import Model

MODEL = "mistral-7b-instruct"

router = APIRouter(
    prefix=f"/{MODEL}/chat/completions",
    tags=[MODEL],
)


MODEL_PATH = "/mnt/nvme1n1p2/huggingface-models/mistralai/Mistral-7B-Instruct-v0.2"

model = Model(model_path=MODEL_PATH)


@router.post("/")
async def chat(chat_payload: ChatPayload):
    chat_dict = chat_payload.model_dump(mode='python')

    response = model.generate_answer(messages=chat_dict['messages'])

    return response
