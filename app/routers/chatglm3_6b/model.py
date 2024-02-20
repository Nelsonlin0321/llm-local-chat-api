from uuid import uuid4
import torch
from transformers import AutoModel, AutoTokenizer


class Model():
    def __init__(self, model_path) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True, local_files_only=True)

        self.model = AutoModel.from_pretrained(model_path,
                                               trust_remote_code=True,device_map="auto").half().eval()

    @torch.no_grad
    def generate_answer(self, messages, model_name="chatglm3-6b"):

        input_massage = messages.pop(-1)['content']
        answer,_ = self.model.chat(self.tokenizer,input_massage, messages)

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
