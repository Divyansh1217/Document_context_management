from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from typing import List
from pydantic import BaseModel

from db import get_db
from models import DocumentRecord
from config import settings
from services import hybrid_search

router = APIRouter(prefix="/chat", tags=["Chat"])

# 🔵 Initialize HuggingFace Embeddings (Groq doesn't provide embeddings)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔵 Initialize Groq Client (REPLACES ChatGoogleGenerativeAI)
groq_client = Groq(api_key=settings.GROQ_API_KEY)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []
    search_type: str = "hybrid"  # hybrid, vector, bm25

@router.post("/", response_model=dict)
async def chat_with_database(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Search using hybrid search (vector + BM25) and answer using Groq + Llama 3."""
    # Use hybrid search
    top_docs = await hybrid_search(request.query, top_k=2)

    if not top_docs:
        return {"answer": "I don't have any documents in my database yet.", "sources": []}

    context_text = ""
    sources = []
    for doc in top_docs:
        context_text += f"\n\n--- Source: {doc.document_name} ---\n{doc.content_text}\n"
        sources.append(doc.document_name)

    system_prompt = f"""
You are an expert AI data assistant. 
You have been provided with full document texts below.
Your job is to scan the text carefully and find the exact answer to the user's query.

RULES:
1. Answer using ONLY the provided DATABASE CONTEXT.
2. Do not hallucinate or use outside knowledge.
3. If the answer is not in the text, state "I cannot find the exact answer in the database documents."

DATABASE CONTEXT:
{context_text}
"""

    # 🔵 Format messages for Groq API (different from LangChain format)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.query})

    # 🔵 Call Groq API directly (REPLACES LangChain chain.invoke())
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
        temperature=0
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(sources)),
        "search_type": request.search_type
    }