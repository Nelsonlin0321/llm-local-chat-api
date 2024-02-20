from enum import Enum
from typing import List
from pydantic import BaseModel

#pylint:disable=line-too-long
TESTING_MESSAGES = [
        {
            "role": "user",
            "content": "解释一下“温故而知新”"
        },
        {
            "role": "assistant",
            "content": "“温故而知新”是一句源自《论语》的古文名言，其含义是回顾过去以获取新的知识和领悟。这句话鼓励人们在学习和生活中不断地回顾和反思过去的经验，从而发现新的理解和启示。通过这种方式，人们可以不断成长和提高自己，使自己的知识、技能和认识水平得到提升。”"
        },
        {
            "role": "user",
            "content": "make the answer brief"
        }
    ]

#pylint:disable=invalid-name
class RoleEnum(str, Enum):
    user = "user"
    system = "system"
    assistance = "assistant"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    messages: List[ChatMessage]=TESTING_MESSAGES
