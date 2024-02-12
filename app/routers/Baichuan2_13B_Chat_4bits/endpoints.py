from uuid import uuid4
from fastapi import APIRouter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from .payload import ChatPayload

MODEL  = "Baichuan2-13B-Chat-4bits"

router = APIRouter(
    prefix=f"/{MODEL}/chat/completions",
    tags=[MODEL],
)


MODEL_PATH = "/mnt/nvme1n1p2/huggingface-models/baichuan-inc-Baichuan2-13B-Chat-4bits"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
    revision="v2.0",
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    revision="v2.0",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True)

model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH,
                                                           revision="v2.0",
                                                           local_files_only=True)

@router.post("/")
async def chat(chat_payload:ChatPayload):

    chat_dict  = chat_payload.model_dump(mode='python')

    answer = model.chat(tokenizer, chat_dict['messages'])
    response = {
        "id": str(uuid4()),
        "object": "chat.completion",
        "created": None,
        "model": chat_payload.model,
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "logprobs": None,
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }

    return response