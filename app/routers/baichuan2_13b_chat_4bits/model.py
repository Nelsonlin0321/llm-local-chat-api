import torch
from uuid import uuid4
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


class Model():
    def __init__(self, model_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       revision="v2.0",
                                                       use_fast=False,
                                                       trust_remote_code=True,
                                                       local_files_only=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision="v2.0",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True).eval()

        self.model.generation_config = GenerationConfig.from_pretrained(
            model_path,
            revision="v2.0",
            local_files_only=True)
        

    @torch.no_grad
    def generate_answer(self, messages, model_name="Baichuan2-13B"):

        answer = self.model.chat(self.tokenizer, messages)

        response = {
            "id": str(uuid4()),
            "object": "chat.completion",
            "created": None,
            "model": model_name,
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
