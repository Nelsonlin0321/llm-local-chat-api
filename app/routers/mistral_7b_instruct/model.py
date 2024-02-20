import re
from uuid import uuid4
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_gpu_with_most_memory():
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print(" GPUs available.")
        return None

    min_memory_allocated=0
    max_memory_device = None

    for i in range(device_count):
        device = torch.device(f"cuda:{i}")
        memory_allocated = torch.cuda.memory_allocated(device)
        if min_memory_allocated >= memory_allocated:
            min_memory_allocated = memory_allocated
            max_memory_device = device

    return max_memory_device

class Model():
    def __init__(self, model_path) -> None:
        
        self.device = get_gpu_with_most_memory()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device).half().eval()

    @torch.no_grad
    def generate_answer(self, messages, model_name="Mistral-7B-Instruct-v0.2"):

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=32*1000, 
                                            do_sample=True,pad_token_id=self.tokenizer.eos_token_id)

        total_tokens = generated_ids.shape[1]

        decoded = self.tokenizer.batch_decode(generated_ids)

        response = decoded[0]

        result = re.findall(r'</s>(.*?)</s>', response)[0]

        answer = result.split("[/INST]")[1].strip()

        response = {
            "id": str(uuid4()),
            "object": "chat.completion",
            "created": None,
            "model": model_name,
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": total_tokens
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
