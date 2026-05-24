import asyncio
import json
import uuid
import os
import smtplib
import tempfile
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from twilio.rest import Client as TwilioClient
from rank_bm25 import BM25Okapi
from config import settings
from db import AsyncSessionLocal
import models

# 🔵 Initialize Groq Client (REPLACES Google GenAI)
groq_client = Groq(api_key=settings.GROQ_API_KEY)

# 🔵 Initialize HuggingFace Embeddings (Groq doesn't provide embeddings)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Qdrant Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "unidocs_chunks"
qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=None)

# BM25 Index (in-memory, rebuild on startup)
bm25_index = None
document_corpus = []
document_ids = []

async def initialize_bm25_index():
    """Initialize BM25 index with all documents."""
    global bm25_index, document_corpus, document_ids
    
    async with AsyncSessionLocal() as db:
        from sqlalchemy.future import select
        result = await db.execute(select(models.DocumentRecord))
        docs = result.scalars().all()
        
        document_corpus = [doc.content_text.split() for doc in docs]
        document_ids = [doc.id for doc in docs]
        
        if document_corpus:
            bm25_index = BM25Okapi(document_corpus)
            print(f"[BM25] Index initialized with {len(document_corpus)} documents")

async def perform_ml_extraction(text: str) -> list[str]:
    """Extract important chunks from document text using Groq + Llama 3."""
    prompt = f"""
You are an expert data extraction system.
Review the following document and extract the most important, high-value chunks of information.
Focus on key facts, rules, core concepts, or critical clauses.
Do NOT summarize; extract the actual concepts or verbatim crucial lines.
Return the output STRICTLY as a JSON array of strings. 
Example format: ["Important fact 1", "Important clause 2"]

Document text:
---
{text}
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 🔵 Using Groq instead of Google GenAI
            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            extracted_chunks = json.loads(response.choices[0].message.content)
            if isinstance(extracted_chunks, list):
                return extracted_chunks
            elif isinstance(extracted_chunks, dict) and "chunks" in extracted_chunks:
                return extracted_chunks["chunks"]
            return [str(extracted_chunks)]
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                print(f"[Worker] ⚠️ Rate limit hit. Sleeping for 15s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(15) 
            else:
                print(f"[Worker] ❌ Groq extraction failed: {e}")
                return []
            
    print("[Worker] ❌ Max retries reached. Extraction failed.")
    return []

async def upsert_chunks_to_qdrant(doc_id: int, successful_extractions: list[str]):
    """Embeds the text and pushes the vectors to Qdrant."""
    points = []
    for chunk_text in successful_extractions:
        # 🔵 Using HuggingFace for embeddings (Groq doesn't provide embeddings)
        embedding = embeddings_model.embed_query(chunk_text)
        
        points.append(
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=embedding, 
                payload={
                    "doc_id": doc_id,         
                    "text_chunk": chunk_text  
                }
            )
        )
    
    await qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

async def hybrid_search(query: str, top_k: int = 3):
    """
    🎯 HYBRID SEARCH: Combines Vector (Semantic) + BM25 (Keyword)
    
    Returns documents ranked by combined score.
    """
    global bm25_index, document_corpus, document_ids
    
    results = {}
    
    # 1. Vector Search (Semantic)
    query_vector = embeddings_model.embed_query(query)
    
    async with AsyncSessionLocal() as db:
        from sqlalchemy.future import select
        from sqlalchemy import text
        
        # pgvector cosine similarity search
        vector_query = select(models.DocumentRecord).order_by(
            models.DocumentRecord.embedding.cosine_distance(query_vector)
        ).limit(top_k * 2)
        
        vector_result = await db.execute(vector_query)
        vector_docs = vector_result.scalars().all()
        
        for i, doc in enumerate(vector_docs):
            if doc.id not in results:
                results[doc.id] = {"doc": doc, "vector_score": 0, "bm25_score": 0}
            # Inverse distance = higher score is better
            results[doc.id]["vector_score"] = 1 - (i + 1) / len(vector_docs)
    
    # 2. BM25 Search (Keyword)
    if bm25_index and document_corpus:
        query_tokens = query.lower().split()
        bm25_scores = bm25_index.get_scores(query_tokens)
        
        for i, score in enumerate(bm25_scores):
            if score > 0 and i < len(document_ids):
                doc_id = document_ids[i]
                if doc_id not in results:
                    # Fetch document if not in vector results
                    doc = await db.get(models.DocumentRecord, doc_id)
                    if doc:
                        results[doc_id] = {"doc": doc, "vector_score": 0, "bm25_score": 0}
                
                if doc_id in results:
                    results[doc_id]["bm25_score"] = score
    
    # 3. Combine Scores (Weighted Hybrid)
    VECTOR_WEIGHT = 0.6
    BM25_WEIGHT = 0.4
    
    for doc_id, data in results.items():
        # Normalize scores
        data["combined_score"] = (
            VECTOR_WEIGHT * data["vector_score"] + 
            BM25_WEIGHT * min(data["bm25_score"] / 10, 1)  # Normalize BM25
        )
    
    # 4. Sort by combined score
    sorted_results = sorted(
        results.values(), 
        key=lambda x: x["combined_score"], 
        reverse=True
    )[:top_k]
    
    return [r["doc"] for r in sorted_results]

async def send_whatsapp_message(to_number: str, message: str, excel_file_path: str = None):
    """Send WhatsApp message via Twilio with optional Excel attachment."""
    def _send():
        client_twilio = TwilioClient(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        recipient = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"
        
        if excel_file_path and os.path.exists(excel_file_path):
            # For WhatsApp, we send a message with a link (direct file upload not supported in sandbox)
            # In production, use WhatsApp Business API media endpoint
            message_obj = client_twilio.messages.create(
                from_=settings.TWILIO_WHATSAPP_NUMBER,
                body=f"{message}\n\n📎 Excel file ready! (Contact admin for file)",
                to=recipient
            )
            return message_obj.sid
        else:
            message_obj = client_twilio.messages.create(
                from_=settings.TWILIO_WHATSAPP_NUMBER,
                body=message,
                to=recipient
            )
            return message_obj.sid
    
    return await asyncio.to_thread(_send)

async def send_gmail_email(to_email: str, subject: str, body: str, excel_file_path: str = None):
    """Send email via Gmail SMTP with optional Excel attachment."""
    def _send():
        msg = MIMEMultipart()
        msg['From'] = settings.GMAIL_SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if excel_file_path and os.path.exists(excel_file_path):
            with open(excel_file_path, 'rb') as f:
                part = MIMEBase('application', 'vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(excel_file_path))
                msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(settings.GMAIL_SENDER_EMAIL, settings.GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
    
    return await asyncio.to_thread(_send)

async def parse_natural_command(text: str):
    """
    🎯 Parse natural language commands like:
    - "create the excel sheet and send it to my whatsapp"
    - "send this document to my email"
    - "create summary for document 5 and whatsapp me"
    
    Returns structured command object.
    """
    prompt = f"""
You are a command parser. Analyze the user's request and extract the intent.

Return a JSON object with these fields:
- "action": one of ["excel", "summary", "chat", "search"]
- "send_whatsapp": boolean
- "send_email": boolean
- "whatsapp_number": string or null
- "email_address": string or null
- "document_id": integer or null
- "query": string (search query or extraction request)
- "custom_message": string or null

User request: "{text}"

Example outputs:
- {{"action": "excel", "send_whatsapp": true, "send_email": false, "whatsapp_number": "+1234567890", "email_address": null, "document_id": null, "query": "extract patient data", "custom_message": null}}
- {{"action": "summary", "send_whatsapp": false, "send_email": true, "whatsapp_number": null, "email_address": "user@gmail.com", "document_id": 5, "query": null, "custom_message": null}}
"""

    # 🔵 Using Groq instead of Google GenAI
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

async def process_document_pipeline(document_id: int, text: str, notify_email: str = None, notify_phone: str = None):
    """Background task orchestration with auto-notification."""
    print(f"[Worker] Started processing Document ID: {document_id}")
    async with AsyncSessionLocal() as db:
        try:
            print("[Worker] Sending to Groq for extraction...")
            important_chunks = await perform_ml_extraction(text)
            
            if not important_chunks:
                raise ValueError("No chunks extracted.")
            
            print(f"[Worker] Extracted {len(important_chunks)} chunks.")
            await upsert_chunks_to_qdrant(document_id, important_chunks)
            
            # Update DB status
            doc = await db.get(models.DocumentRecord, document_id)
            if doc:
                doc.status = "vectorized"
                await db.commit()
                print(f"[Worker] Document {document_id} successfully vectorized.")

            # Auto-notify
            if notify_email:
                await send_gmail_email(
                    notify_email, 
                    "📄 Document Processed", 
                    f"Document ID {document_id} has been successfully extracted & vectorized."
                )
            if notify_phone:
                await send_whatsapp_message(
                    notify_phone, 
                    f"✅ Doc {document_id} processed & vectorized successfully."
                )

        except Exception as e:
            print(f"[Worker] Failed to process Document {document_id}: {e}")
            doc = await db.get(models.DocumentRecord, document_id)
            if doc:
                doc.status = "failed"
                await db.commit()