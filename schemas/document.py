from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DocumentResponse(BaseModel):
    id: int
    filename: str
    status: str
    uploaded_at: datetime

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    document_id: int
    message: str

class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = []

class ChatMessageResponse(BaseModel):
    id: int
    text: str
    is_user: bool
    sources: list[str] = []
    created_at: datetime

    class Config:
        from_attributes = True