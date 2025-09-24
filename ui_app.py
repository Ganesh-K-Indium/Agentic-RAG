# file: ui_app.py
import streamlit as st
import requests
import os
import uuid
from app_logger import log_response  # kept as in your file

RAG_URL = os.getenv("RAG_URL", "http://localhost:8001/ask")

# Load Tailwind CSS
tailwind_cdn = """
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
"""
st.markdown(tailwind_cdn, unsafe_allow_html=True)

# Streamlit page config
st.set_page_config(page_title="Agentic AI Notebook", layout="wide")

# Dark theme global background + typing CSS
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #0f172a; /* slate-900 */
        color: #f1f5f9;           /* slate-100 */
    }

    /* Kill Streamlit’s textarea wrapper */
    div[data-baseweb="textarea"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Custom input box */
    div[data-baseweb="textarea"] textarea {
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Typing dots animation (used in assistant loader) */
    @keyframes blink {
        0% { opacity: .2; }
        20% { opacity: 1; }
        100% { opacity: .2; }
    }
    .dot {
        height: 8px;
        width: 8px;
        background-color: #f3f4f6;
        border-radius: 50%;
        display: inline-block;
        animation: blink 1.4s infinite both;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    </style>
    """,
    unsafe_allow_html=True,
)

# App header
st.markdown(
    """
    <div class="container mx-auto px-6 py-6 text-center">
        <h1 class="text-4xl font-bold text-gray-100 mb-2">
            📓 Secondary Research Agent 
        </h1>
        <p class="text-gray-400 text-lg">
            Your AI-powered knowledge manager
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session
if "messages" not in st.session_state:
    # messages are stored NEWEST-FIRST (index 0 is top)
    st.session_state.messages = []

# User input
user_query = st.chat_input("Ask me anything...")

if user_query:
    # Create an assistant placeholder (loading) and attach the query so we can process it after rendering
    placeholder_id = str(uuid.uuid4())
    assistant_placeholder = {
        "id": placeholder_id,
        "role": "assistant",
        "content": None,
        "loading": True,
        "processing_started": False,
        "user_query": user_query,
    }

    # Insert assistant placeholder first, then insert the user message so the user message is on top
    st.session_state.messages.insert(0, assistant_placeholder)
    st.session_state.messages.insert(0, {"id": str(uuid.uuid4()), "role": "user", "content": user_query})

# Render messages (newest-first, top -> down)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='
                display: flex; 
                justify-content: flex-end; 
                margin-bottom: 1rem;'>
                <div style='
                    max-width: 70%; 
                    background: linear-gradient(135deg, #f87171, #ef4444); 
                    color: white; 
                    padding: 12px 16px; 
                    border-radius: 18px 18px 4px 18px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15); 
                    font-size: 15px;
                    line-height: 1.5;'>
                    {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # assistant
        if msg.get("loading"):
            # typing loader bubble
            st.markdown(
                f"""
                <div style='
                    display: flex;
                    justify-content: flex-start;
                    margin-bottom: 1rem;'>
                    <div style='
                        max-width: 70%;
                        background: #1f2937;
                        color: #f3f4f6;
                        padding: 12px 16px;
                        border-radius: 18px 18px 18px 4px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                        font-size: 15px;
                        line-height: 1.5;
                        display: flex;
                        align-items: center;
                        gap: 8px;'>
                        <span class="dot"></span>
                        <span class="dot"></span>
                        <span class="dot"></span>
                        <span style="opacity:0.7; margin-left:6px; font-size:13px;">Thinking…</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # normal assistant bubble with final content
            content = msg.get("content", "")
            st.markdown(
                f"""
                <div style='
                    display: flex; 
                    justify-content: flex-start; 
                    margin-bottom: 1rem;'>
                    <div style='
                        max-width: 70%; 
                        background: #1f2937; 
                        color: #f3f4f6; 
                        padding: 12px 16px; 
                        border-radius: 18px 18px 18px 4px; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.15); 
                        font-size: 15px;
                        line-height: 1.5;'>
                        {content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# After rendering, find the first assistant placeholder that hasn't started processing and process it.
for msg in st.session_state.messages:
    if msg["role"] == "assistant" and msg.get("loading") and not msg.get("processing_started"):
        # mark started so we don't process it again on reruns
        msg["processing_started"] = True
        try:
            payload = {"query": msg.get("user_query", "")}
            response = requests.post(RAG_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            # Save logs (no change)
            try:
                log_response(payload, data)
            except Exception:
                # don't break UI if logging fails
                pass

            # Extract final assistant answer (same logic as before)
            answer = data.get("answer", {})
            if isinstance(answer, dict) and "logs" in answer:
                source = answer.get("source", "unknown source")
                details = []
                if answer.get("space_key"):
                    details.append(f"space key {answer['space_key']}")
                if answer.get("file_name"):
                    details.append(f"file {answer['file_name']}")
                detail_text = ", ".join(details) if details else ""
                final_text = f"Ingestion completed for {source} {detail_text}".strip()

            elif isinstance(answer, dict) and "Intermediate_message" in answer:
                final_text = answer["Intermediate_message"]

            elif isinstance(answer, str):
                final_text = answer
            else:
                final_text = str(answer)

        except Exception as e:
            final_text = f"❌ Error: {e}"

        # update the assistant message (replace loader with final text)
        msg["content"] = final_text
        msg["loading"] = False
        # we updated state — rerun so the UI refreshes and shows the final answer immediately
        st.rerun()
        # st.experimental_rerun() will stop execution and re-run, so nothing after here runs for this invocation
        break
