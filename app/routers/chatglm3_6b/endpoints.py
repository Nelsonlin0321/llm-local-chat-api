from fastapi import APIRouter
from .payload import ChatPayload
from .model import Model

MODEL = "chatglm3-6b"

router = APIRouter(
    prefix=f"/{MODEL}/chat/completions",
    tags=[MODEL],
)


MODEL_PATH = "/mnt/nvme1n1p2/huggingface-models/THUDM/chatglm3-6b"

model = Model(model_path=MODEL_PATH)


@router.post("/")
async def chat(chat_payload: ChatPayload):

    chat_dict = chat_payload.model_dump(mode='python')

    response = model.generate_answer(
        model_name=chat_payload.model, messages=chat_dict['messages'])

    return response
