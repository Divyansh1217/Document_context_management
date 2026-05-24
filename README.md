# 📄 AI Document Assistant

A powerful document processing system that uses **AI (Groq/Gemini)** to extract structured data from documents, store them in a **vector database**, and enable **semantic search + keyword search (BM25)**. Extract data to Excel, chat with documents, and send results via **WhatsApp** or **Gmail**.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📊 **Excel Extraction** | Extract structured data from documents to Excel sheets |
| 💬 **Document Chat** | Chat with your documents using AI-powered Q&A |
| 🔍 **Hybrid Search** | Combines Vector (Semantic) + BM25 (Keyword) search |
| 📱 **WhatsApp Notifications** | Send Excel sheets and summaries directly to WhatsApp |
| 📧 **Gmail Integration** | Email extracted data with Excel attachments |
| 🎯 **Natural Language Commands** | Type commands like *"create excel and send to my whatsapp"* |
| 🔄 **Background Processing** | Async document vectorization with auto-notifications |

---

## 📁 Project Structure
document_multi_agent/
├── routers/
│ ├── init.py
│ ├── documents.py # Document upload & management
│ ├── chat.py # Document Q&A chat
│ ├── export.py # Excel extraction & sharing
│ ├── notifications.py # WhatsApp & Gmail notifications
│ └── commands.py # Natural language command execution
├── services.py # Core AI & notification services
├── models.py # SQLAlchemy database models
├── db.py # Database configuration
├── config.py # Application settings
├── main.py # FastAPI application entry point
├── app.py # Streamlit frontend
├── .env # Environment variables (DO NOT COMMIT)
├── .gitignore # Git ignore rules
├── requirements.txt # Python dependencies
└── README.md # This file



create .env file

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/yourdb

# AI API Keys
GEMINI_API_KEY=your_gemini_api_key
# OR
GROQ_API_KEY=your_groq_api_key

# Twilio WhatsApp
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# Gmail
GMAIL_SENDER_EMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=your_16_char_gmail_app_password




/api/v1/documents/upload/
POST
Upload and vectorize a document

/api/v1/documents/list/
GET
List all uploaded documents

/api/v1/documents/{id}/
GET
Get specific document by ID

/api/v1/chat/
POST
Chat with documents (Hybrid Search)

/api/v1/export/excel/
POST
Extract data to Excel

/api/v1/export/excel-and-share/
POST
Extract Excel + Send to WhatsApp/Gmail

/api/v1/commands/execute/
POST
Execute natural language commands

/api/v1/notifications/whatsapp/
POST
Send WhatsApp message

/api/v1/notifications/email/
POST
Send Gmail email