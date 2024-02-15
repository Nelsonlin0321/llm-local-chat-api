from typing import Optional
from pydantic import BaseModel


class RetrievalLoad(BaseModel):
    context: Optional[str] = None
    question: str
    file_name: str
