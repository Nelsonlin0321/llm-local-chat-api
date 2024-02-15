from typing import Optional
from pydantic import BaseModel


class RetrievalLoad(BaseModel):
    context: Optional[str] = None
    question: str = "在自动研磨模式下，如何保护生产板？"
    file_name: str="关于精密研磨优化改进制作说明.docx"
