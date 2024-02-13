from fastapi import APIRouter
from .payload import ChatPayload
from .model import Model

MODEL = "Baichuan2-13B-Chat-4bits"

router = APIRouter(
    prefix=f"/{MODEL}/chat/completions",
    tags=[MODEL],
)


MODEL_PATH = "/mnt/nvme1n1p2/huggingface-models/baichuan-inc-Baichuan2-13B-Chat-4bits"

model = Model(model_path=MODEL_PATH)


@router.post("/")
async def chat(chat_payload: ChatPayload):

    chat_dict = chat_payload.model_dump(mode='python')

    response = model.generate_answer(
        model_name=chat_payload.model, messages=chat_dict['messages'])

    return response
