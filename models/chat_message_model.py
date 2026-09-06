from sqlalchemy import Column, Integer, ForeignKey, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from database import Base

class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=False)
    is_user = Column(Boolean, nullable=False)
    sources = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())