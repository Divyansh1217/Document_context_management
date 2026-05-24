import streamlit as st
import pandas as pd
import io
import json
import requests
import os
from groq import Groq

# 🔵 Initialize Groq Client (REPLACES Google GenAI)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found. Please set the GROQ_API_KEY environment variable.")
    groq_client = None
else:
    groq_client = Groq(api_key=GROQ_API_KEY)

# FastAPI Backend URL
API_URL = "http://localhost:8001"

st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("📄 AI Document Assistant")
st.markdown("Upload a document to extract structured data to Excel, or chat with it to find specific answers. **Powered by Groq + Llama 3**")

# --- Session State Management ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- 1. File Upload Section ---
uploaded_file = st.file_uploader("1. Upload a text document (.txt, .md)", type=["txt", "md"])
doc_text = ""

if uploaded_file:
    doc_text = uploaded_file.getvalue().decode("utf-8")
    
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.chat_messages = []
        st.session_state.current_file = uploaded_file.name
    
    with st.expander("Preview Uploaded Document", expanded=False):
        st.text(doc_text[:1000] + "\n\n... [Document Truncated for Preview]")
    
    # Upload with notification options
    st.subheader("📤 Upload with Auto-Notification")
    col1, col2 = st.columns(2)
    with col1:
        notify_email = st.text_input("Email for notification (optional)")
    with col2:
        notify_phone = st.text_input("WhatsApp for notification (optional)")
    
    if st.button("Save to Database & Notify", type="secondary"):
        with st.spinner("Vectorizing and saving to the database..."):
            payload = {
                "drive_link": "local-upload", 
                "doc_name": uploaded_file.name,
                "full_text": doc_text,
                "notify_email": notify_email if notify_email else None,
                "notify_phone": notify_phone if notify_phone else None
            }
            try:
                res = requests.post(f"{API_URL}/upload-document/", json=payload)
                if res.status_code == 200:
                    st.success("✅ Document successfully uploaded and vectorized!")
                else:
                    st.error(f"Backend Error {res.status_code}: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Failed to connect to the backend. Is FastAPI running?")

st.divider()

# --- 2. Feature Tabs ---
tab1, tab2 = st.tabs(["📊 Extract Data to Excel", "💬 Chat with Document"])

# ==========================================
# TAB 1: EXCEL EXTRACTION
# ==========================================
with tab1:
    st.header("Extract Structured Data")
    extraction_query = st.text_input(
        "What data do you want to extract into columns?",
        placeholder="e.g., Extract all patient names, their ages, the drug administered, and any side effects."
    )
    
    if st.button("Extract & Generate Excel", type="primary"):
        if not groq_client:
            st.warning("Cannot extract: Groq API key is missing.")
        elif not uploaded_file:
            st.warning("Please upload a document first.")
        elif not extraction_query:
            st.warning("Please tell the LLM what to extract.")
        else:
            with st.spinner("Analyzing document and structuring the data..."):
                try:
                    system_prompt = """
                    You are an expert data extraction AI. Extract information from the provided document 
                    exactly as requested by the user.
                    
                    You MUST return the data as a JSON object containing a single key called "data". 
                    The value of "data" must be a list of objects, where each object represents one row of data.
                    The keys inside the objects should be the column names derived from the user's query.
                    If a specific piece of data is missing for a row, use "N/A".
                    """
                    
                    # 🔵 Using Groq SDK to generate structured JSON content
                    response = groq_client.chat.completions.create(
                        model='llama-3.1-70b-versatile',
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that outputs valid JSON only."},
                            {"role": "user", "content": f"User Request: {extraction_query}\n\nDocument Text:\n{doc_text}\n\n{system_prompt}"}
                        ],
                        temperature=0,
                        response_format={"type": "json_object"}
                    )

                    extracted_json = json.loads(response.choices[0].message.content)
                    data_list = extracted_json.get("data", [])

                    if not data_list:
                        st.warning("No matching data found in the document.")
                    else:
                        df = pd.DataFrame(data_list)
                        st.success("Extraction Complete!")
                        st.dataframe(df, use_container_width=True)

                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name='Extracted_Data')
                            worksheet = writer.sheets['Extracted_Data']
                            for i, col in enumerate(df.columns):
                                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                                worksheet.set_column(i, i, min(max_len, 50))
                            
                            excel_buffer.seek(0)

                        st.download_button(
                            label="📥 Download Excel File",
                            data=excel_buffer,
                            file_name="ai_extracted_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ==========================================
# TAB 2: DOCUMENT Q&A (CHAT)
# ==========================================
with tab2:
    st.header("💬 Chat with All Documents")
    st.markdown("Ask a question, and the AI will search the entire SQL database to find the answer.")
    
    if "global_chat_messages" not in st.session_state:
        st.session_state.global_chat_messages = []

    for message in st.session_state.global_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the database..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.global_chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching the database and reading documents..."):
                try:
                    payload = {
                        "query": prompt,
                        "history": st.session_state.global_chat_messages[:-1] 
                    }
                    
                    res = requests.post(f"{API_URL}/chat/", json=payload)
                    
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        
                        st.markdown(answer)
                        
                        if sources:
                            st.caption(f"**Sources used:** {', '.join(sources)}")
                        
                        st.session_state.global_chat_messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"Backend Error {res.status_code}: {res.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Failed to connect to the backend. Is FastAPI running?")