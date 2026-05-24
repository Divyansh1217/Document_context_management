from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from pydantic import BaseModel

from db import get_db
from models import DocumentRecord
from config import settings
from services import hybrid_search

router = APIRouter(prefix="/chat", tags=["Chat"])

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=settings.GEMINI_API_KEY
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []
    search_type: str = "hybrid"  # hybrid, vector, bm25

@router.post("/", response_model=dict)
async def chat_with_database(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Search using hybrid search (vector + BM25) and answer using Gemini."""
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

    api_messages = [("system", system_prompt)]
    for msg in request.history:
        api_messages.append((msg.role, msg.content))
    api_messages.append(("human", request.query))

    prompt = ChatPromptTemplate.from_messages(api_messages)
    chain = prompt | llm
    response = chain.invoke({})

    return {
        "answer": response.content,
        "sources": list(set(sources)),
        "search_type": request.search_type
    }