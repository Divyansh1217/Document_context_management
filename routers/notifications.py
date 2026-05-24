from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import asyncio
import os
import tempfile
from typing import Optional
from pydantic import BaseModel

from config import settings
from services import send_whatsapp_message, send_gmail_email

router = APIRouter(prefix="/notifications", tags=["Notifications"])

class WhatsAppRequest(BaseModel):
    to_number: str
    message: str

class EmailRequest(BaseModel):
    to_email: str
    subject: str
    body: str

@router.post("/whatsapp/", response_model=dict)
async def send_whatsapp(req: WhatsAppRequest, excel_file: Optional[UploadFile] = File(None)):
    """Send WhatsApp message with optional Excel attachment."""
    temp_file_path = None
    
    if excel_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_file_path = temp_file.name
        temp_file.close()
        
        content = await excel_file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)

    try:
        sid = await send_whatsapp_message(
            to_number=req.to_number,
            message=req.message,
            excel_file_path=temp_file_path
        )
        return {"status": "success", "message_sid": sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WhatsApp failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.post("/email/", response_model=dict)
async def send_email(
    to_email: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    excel_file: Optional[UploadFile] = File(None)
):
    """Send email via Gmail SMTP with optional Excel attachment."""
    temp_file_path = None
    
    if excel_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_file_path = temp_file.name
        temp_file.close()
        
        content = await excel_file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)

    try:
        await send_gmail_email(
            to_email=to_email,
            subject=subject,
            body=body,
            excel_file_path=temp_file_path
        )
        return {"status": "success", "message": "Email sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)