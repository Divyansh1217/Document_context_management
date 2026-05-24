from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import pandas as pd
import io
import json
from typing import List

from db import get_db
from models import DocumentRecord
from config import settings
from services import process_document_pipeline
import asyncio

app = FastAPI(title="Document Vectorization & Excel Extraction API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔵 Initialize HuggingFace Embeddings (Groq doesn't provide embeddings)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔵 Initialize Groq Client (REPLACES ChatGoogleGenerativeAI)
groq_client = Groq(api_key=settings.GROQ_API_KEY)

# --- PYDANTIC MODELS FOR STRUCTURED LLM OUTPUT ---
class ExtractedData(BaseModel):
    subject_id: str = Field(description="The unique identifier for the subject or patient")
    trial_phase: str = Field(description="The phase of the clinical trial (e.g., Phase I, Phase II)")
    compliance_status: str = Field(description="Current regulatory compliance status")
    key_findings: str = Field(description="Brief summary of findings or notes")

class DataExtractionList(BaseModel):
    records: List[ExtractedData]

class SearchRequest(BaseModel):
    query: str

class DocumentUploadRequest(BaseModel):
    drive_link: str
    doc_name: str
    full_text: str
    notify_email: str = None
    notify_phone: str = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []

@app.post("/upload-document/")
async def upload_and_vectorize(request: DocumentUploadRequest, db: AsyncSession = Depends(get_db)):
    """Receives text, vectorizes the ENTIRE document, and saves to pgvector."""
    try:
        vector = embeddings_model.embed_query(request.full_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Embedding failed: {e}")
    
    new_doc = DocumentRecord(
        drive_link=request.drive_link,
        document_name=request.doc_name,
        content_text=request.full_text,
        embedding=vector,
        status="pending"
    )

    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    # Trigger background processing
    asyncio.create_task(process_document_pipeline(
        new_doc.id, 
        request.full_text,
        request.notify_email,
        request.notify_phone
    ))

    return {"status": "success", "doc_id": new_doc.id}

@app.post("/chat/")
async def chat_with_database(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Searches the pgvector database and answers the query using Groq + Llama 3."""
    # 1. Embed the user's question
    query_vector = embeddings_model.embed_query(request.query)
    
    # 2. Retrieve the Top 2 most relevant FULL documents
    query = select(DocumentRecord).order_by(
        DocumentRecord.embedding.cosine_distance(query_vector)
    ).limit(2)

    result = await db.execute(query)
    top_docs = result.scalars().all()

    if not top_docs:
        return {"answer": "I don't have any documents in my database yet.", "sources": []}

    # 3. Combine the massive text of the retrieved documents
    context_text = ""
    sources = []
    for doc in top_docs:
        context_text += f"\n\n--- Source: {doc.document_name} ---\n{doc.content_text}\n"
        sources.append(doc.document_name)

    # 4. Build system prompt
    system_prompt = f"""
You are an expert AI data assistant. 
You have been provided with full, massive document texts below.
Your job is to scan the text carefully and find the exact answer to the user's query.

RULES:
1. Answer using ONLY the provided DATABASE CONTEXT.
2. Do not hallucinate or use outside knowledge.
3. If the specific answer is not contained anywhere within the text, explicitly state "I cannot find the exact answer in the database documents."

DATABASE CONTEXT:
{context_text}
"""

    # 5. Format messages including chat history
    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.query})

    # 6. Get the LLM response from Groq
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
        temperature=0
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(sources))
    }

@app.post("/search-and-export/")
async def search_and_export_to_excel(request: SearchRequest, db: AsyncSession = Depends(get_db)):
    """Vector search for document, extract data, return Excel file."""
    # 1. Embed the user's search query
    query_vector = embeddings_model.embed_query(request.query)
    
    # 2. Query pgvector for the closest document
    query = select(DocumentRecord).order_by(
        DocumentRecord.embedding.cosine_distance(query_vector)
    ).limit(1)

    result = await db.execute(query)
    closest_doc = result.scalars().first()

    if not closest_doc:
        raise HTTPException(status_code=404, detail="No documents found.")

    # 3. Instruct the LLM to extract structured data
    prompt = f"""
You are an expert data extraction algorithm. Scan the entire provided document text and extract all relevant requested entities.
Extract the trial data from the following document. If a field is missing, output 'N/A'.
Return the output as a JSON object with a "records" key containing a list of objects.

Document Text:
{closest_doc.content_text}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    extracted_json = json.loads(response.choices[0].message.content)
    
    # Handle different JSON structures
    if "records" in extracted_json:
        data_list = extracted_json["records"]
    elif "data" in extracted_json:
        data_list = extracted_json["data"]
    else:
        data_list = [extracted_json]

    # 4. Convert to Pandas DataFrame
    df = pd.DataFrame(data_list)

    # 5. Write to Excel in memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Extracted_Data')

    excel_buffer.seek(0)

    # 6. Return downloadable file
    headers = {
        'Content-Disposition': f'attachment; filename="{closest_doc.document_name}_extraction.xlsx"'
    }

    return StreamingResponse(
        excel_buffer, 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
        headers=headers
    )

@app.get("/")
async def root():
    return {
        "message": "Document Vectorization API is running (Powered by Groq)",
        "version": settings.PROJECT_VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}