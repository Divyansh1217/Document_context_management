from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
import json

from db import get_db
from services import parse_natural_command, send_whatsapp_message, send_gmail_email, hybrid_search
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import pandas as pd
import io
import tempfile
import os

from config import settings

router = APIRouter(prefix="/commands", tags=["Natural Language Commands"])

# 🔵 Initialize HuggingFace Embeddings (Groq doesn't provide embeddings)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔵 Initialize Groq Client (REPLACES ChatGoogleGenerativeAI)
groq_client = Groq(api_key=settings.GROQ_API_KEY)

class NaturalCommandRequest(BaseModel):
    command: str
    whatsapp_number: Optional[str] = None
    email_address: Optional[str] = None

class NaturalCommandResponse(BaseModel):
    status: str
    action: str
    message: str
    notifications_sent: list
    errors: Optional[list] = None

@router.post("/execute/", response_model=dict)
async def execute_natural_command(request: NaturalCommandRequest, db: AsyncSession = Depends(get_db)):
    """
    🎯 NATURAL LANGUAGE COMMAND EXECUTION
    
    Examples:
    - "for this document create the excel sheet and send it to my whatsapp"
    - "create summary for document 5 and email me"
    - "search for clinical trials and send results to +1234567890"
    """
    # Step 1: Parse the natural language command
    parsed = await parse_natural_command(request.command)
    
    # Override with provided contact info if available
    if request.whatsapp_number:
        parsed["whatsapp_number"] = request.whatsapp_number
    if request.email_address:
        parsed["email_address"] = request.email_address
    
    notifications_sent = []
    errors = []
    message = ""
    
    # Step 2: Execute based on action
    if parsed["action"] == "excel":
        # Find document
        if parsed.get("document_id"):
            from models import DocumentRecord
            doc = await db.get(DocumentRecord, parsed["document_id"])
        else:
            top_docs = await hybrid_search(parsed.get("query", "document"), top_k=1)
            doc = top_docs[0] if top_docs else None
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 🔵 Extract data using Groq (REPLACES LangChain structured output)
        prompt = f"""
You are an expert data extraction algorithm. Extract all relevant entities from the document.
Return the output as a JSON object with a "records" key containing a list of objects.

Document Text:
{doc.content_text}
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
        
        # Create Excel
        df = pd.DataFrame(data_list)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_file_path = temp_file.name
        temp_file.close()
        with pd.ExcelWriter(temp_file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Extracted_Data')
        
        message = f"✅ Excel created for: {doc.document_name}\nRecords: {len(data_list)}"
        
        # Send notifications
        if parsed.get("send_whatsapp") and parsed.get("whatsapp_number"):
            try:
                await send_whatsapp_message(
                    to_number=parsed["whatsapp_number"],
                    message=f"{message}\n\n{parsed.get('custom_message', '')}",
                    excel_file_path=temp_file_path
                )
                notifications_sent.append("whatsapp")
            except Exception as e:
                errors.append(f"WhatsApp: {str(e)}")
        
        if parsed.get("send_email") and parsed.get("email_address"):
            try:
                await send_gmail_email(
                    to_email=parsed["email_address"],
                    subject="📊 Excel Extraction",
                    body=f"{message}\n\n{parsed.get('custom_message', '')}",
                    excel_file_path=temp_file_path
                )
                notifications_sent.append("email")
            except Exception as e:
                errors.append(f"Email: {str(e)}")
        
        os.unlink(temp_file_path)
    
    elif parsed["action"] == "summary":
        # Find document
        if parsed.get("document_id"):
            from models import DocumentRecord
            doc = await db.get(DocumentRecord, parsed["document_id"])
        else:
            top_docs = await hybrid_search(parsed.get("query", "document"), top_k=1)
            doc = top_docs[0] if top_docs else None
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 🔵 Generate summary using Groq (REPLACES LangChain chain)
        prompt = f"""
Summarize this document in 5-7 bullet points highlighting the most critical information.

Document Text:
{doc.content_text}
"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        summary_content = response.choices[0].message.content
        
        message = f"📄 Summary for: {doc.document_name}\n\n{summary_content}"
        
        # Send notifications
        if parsed.get("send_whatsapp") and parsed.get("whatsapp_number"):
            try:
                await send_whatsapp_message(
                    to_number=parsed["whatsapp_number"],
                    message=message,
                    excel_file_path=None
                )
                notifications_sent.append("whatsapp")
            except Exception as e:
                errors.append(f"WhatsApp: {str(e)}")
        
        if parsed.get("send_email") and parsed.get("email_address"):
            try:
                await send_gmail_email(
                    to_email=parsed["email_address"],
                    subject="📄 Document Summary",
                    body=message,
                    excel_file_path=None
                )
                notifications_sent.append("email")
            except Exception as e:
                errors.append(f"Email: {str(e)}")
    
    elif parsed["action"] == "search":
        top_docs = await hybrid_search(parsed.get("query", ""), top_k=3)
        message = f"🔍 Found {len(top_docs)} documents:\n\n"
        for doc in top_docs:
            message += f"📄 {doc.document_name} (ID: {doc.id})\n"
        
        if parsed.get("send_whatsapp") and parsed.get("whatsapp_number"):
            try:
                await send_whatsapp_message(
                    to_number=parsed["whatsapp_number"],
                    message=message,
                    excel_file_path=None
                )
                notifications_sent.append("whatsapp")
            except Exception as e:
                errors.append(f"WhatsApp: {str(e)}")
        
        if parsed.get("send_email") and parsed.get("email_address"):
            try:
                await send_gmail_email(
                    to_email=parsed["email_address"],
                    subject="🔍 Search Results",
                    body=message,
                    excel_file_path=None
                )
                notifications_sent.append("email")
            except Exception as e:
                errors.append(f"Email: {str(e)}")
    
    return {
        "status": "success",
        "action": parsed["action"],
        "message": message,
        "notifications_sent": notifications_sent,
        "errors": errors if errors else None,
        "parsed_command": parsed
    }