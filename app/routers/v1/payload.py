from enum import Enum
from typing import List
from pydantic import BaseModel

#pylint:disable=invalid-name
class RoleEnum(str, Enum):
    user = "user"
    system = "system"
    assistance = "assistance"

class ChatMessage(BaseModel):
    role: RoleEnum
    content: str

class ChatPayload(BaseModel):
    model:str = "Baichuan2-13B"
    messages: List[ChatMessage]=[{"role": "user", "content": "解释一下“温故而知新”"}]
