from fastapi import APIRouter, HTTPException
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

VERSION  = "v1"

router = APIRouter(
    prefix=f"/{VERSION}/chat/completions",
    tags=[VERSION],
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
async def chat():
    messages = []
    messages.append({"role": "user", "content": "解释一下“温故而知新”"})
    response = model.chat(tokenizer, messages)
    return response



