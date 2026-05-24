from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from db import Base

class DocumentRecord(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    drive_link = Column(String, nullable=False)
    document_name = Column(String, nullable=False)
    content_text = Column(Text, nullable=False)
    embedding = Column(Vector(384))
    status = Column(String, default="pending", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())