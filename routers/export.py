from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import pandas as pd
import io
import json
import tempfile
import os
from typing import List, Optional
from pydantic import BaseModel, Field

from db import get_db
from models import DocumentRecord
from config import settings
from services import send_whatsapp_message, send_gmail_email, hybrid_search

router = APIRouter(prefix="/export", tags=["Export"])

# 🔵 Initialize HuggingFace Embeddings (Groq doesn't provide embeddings)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔵 Initialize Groq Client (REPLACES ChatGoogleGenerativeAI)
groq_client = Groq(api_key=settings.GROQ_API_KEY)

class ExtractedData(BaseModel):
    subject_id: str = Field(description="Unique identifier for the subject or patient")
    trial_phase: str = Field(description="Phase of the clinical trial")
    compliance_status: str = Field(description="Current regulatory compliance status")
    key_findings: str = Field(description="Brief summary of findings")

class DataExtractionList(BaseModel):
    records: List[ExtractedData]

class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"  # hybrid, vector, bm25

class ExportAndShareRequest(BaseModel):
    query: str
    document_id: Optional[int] = None
    send_whatsapp: bool = False
    whatsapp_number: Optional[str] = None
    send_email: bool = False
    email_address: Optional[str] = None
    email_subject: str = "📊 AI Document Extraction Results"
    email_body: str = "Please find the extracted data attached."
    custom_message: Optional[str] = None

@router.post("/excel/", response_class=StreamingResponse)
async def search_and_export_to_excel(request: SearchRequest, db: AsyncSession = Depends(get_db)):
    """Vector search for document, extract data, return Excel file."""
    # Use hybrid search
    top_docs = await hybrid_search(request.query, top_k=1)
    
    if not top_docs:
        raise HTTPException(status_code=404, detail="No documents found.")
    
    closest_doc = top_docs[0]

    # 🔵 Extract data using Groq (REPLACES LangChain structured output)
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

    df = pd.DataFrame(data_list)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Extracted_Data')

    excel_buffer.seek(0)

    headers = {
        'Content-Disposition': f'attachment; filename="{closest_doc.document_name}_extraction.xlsx"'
    }

    return StreamingResponse(
        excel_buffer, 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
        headers=headers
    )

@router.post("/excel-and-share/", response_model=dict)
async def export_excel_and_share(request: ExportAndShareRequest, db: AsyncSession = Depends(get_db)):
    """
    🎯 MAIN ROUTE: Extract Excel AND send via WhatsApp/Gmail in one call!
    """
    # Step 1: Find the document using hybrid search
    if request.document_id:
        closest_doc = await db.get(DocumentRecord, request.document_id)
        if not closest_doc:
            raise HTTPException(status_code=404, detail="Document not found.")
    else:
        top_docs = await hybrid_search(request.query, top_k=1)
        if not top_docs:
            raise HTTPException(status_code=404, detail="No documents found.")
        closest_doc = top_docs[0]

    # 🔵 Step 2: Extract data using Groq (REPLACES LangChain structured output)
    prompt = f"""
You are an expert data extraction algorithm. Extract all relevant entities from the document.
Extract the data from the following document. If a field is missing, output 'N/A'.
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

    df = pd.DataFrame(data_list)

    # Step 3: Save Excel to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    temp_file_path = temp_file.name
    temp_file.close()

    with pd.ExcelWriter(temp_file_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Extracted_Data')

    notifications_sent = []
    errors = []
    custom_msg = request.custom_message or f"📊 Your document extraction is ready!\n\nDocument: {closest_doc.document_name}\n\nRecords extracted: {len(data_list)}"

    # Step 4: Send via WhatsApp if requested
    if request.send_whatsapp and request.whatsapp_number:
        try:
            await send_whatsapp_message(
                to_number=request.whatsapp_number,
                message=custom_msg,
                excel_file_path=temp_file_path
            )
            notifications_sent.append("whatsapp")
        except Exception as e:
            errors.append(f"WhatsApp: {str(e)}")

    # Step 5: Send via Gmail if requested
    if request.send_email and request.email_address:
        try:
            await send_gmail_email(
                to_email=request.email_address,
                subject=request.email_subject,
                body=request.email_body + f"\n\nDocument: {closest_doc.document_name}\nRecords extracted: {len(data_list)}",
                excel_file_path=temp_file_path
            )
            notifications_sent.append("email")
        except Exception as e:
            errors.append(f"Email: {str(e)}")

    # Step 6: Clean up temp file
    try:
        os.unlink(temp_file_path)
    except:
        pass

    return {
        "status": "success",
        "document": closest_doc.document_name,
        "records_extracted": len(data_list),
        "notifications_sent": notifications_sent,
        "errors": errors if errors else None,
        "search_type": "hybrid"
    }