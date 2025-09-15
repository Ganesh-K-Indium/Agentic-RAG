import requests
import json
from dotenv import load_dotenv
import os
import datetime

# API endpoint
url = "http://127.0.0.1:8000/ask"
payload = {
    "query": "What do you know about OPENAI and its founders?",
}

# --- Formatter function ---
def format_graph_output(data: dict) -> str:
    """Format final graph output into Markdown with clear headings."""
    lines = []
    
    # Access the 'answer' key which contains the actual response data
    answer_data = data.get('answer', {})
    
    # Messages
    if "messages" in answer_data:
        lines.append("## Messages")
        messages = answer_data["messages"]
        if isinstance(messages, list):
            for i, msg in enumerate(messages, 1):
                # Handle message objects with content attribute
                if isinstance(msg, dict):
                    content = msg.get('content', '')
                    msg_type = msg.get('type', 'unknown')
                    lines.append(f"### Message {i} ({msg_type})")
                    lines.append(content)
                    lines.append("")
                else:
                    lines.append(f"- **Message {i}:** {str(msg)}")
        else:
            lines.append(str(messages))
        lines.append("")
    
    # Intermediate message
    if "Intermediate_message" in answer_data:
        lines.append("## Intermediate Message")
        lines.append(answer_data["Intermediate_message"])
        lines.append("")
    
    # Documents
    if "documents" in answer_data:
        lines.append("## Documents")
        documents = answer_data["documents"]
        if isinstance(documents, list):
            for i, doc in enumerate(documents, 1):
                lines.append(f"### Document {i}")
                if isinstance(doc, dict):
                    # Handle document metadata
                    if 'metadata' in doc:
                        lines.append(f"**Metadata:** {json.dumps(doc['metadata'], indent=2)}")
                    # Handle page content
                    if 'page_content' in doc:
                        lines.append(f"**Content:** {doc['page_content']}")
                    # Handle document type
                    if 'type' in doc:
                        lines.append(f"**Type:** {doc['type']}")
                else:
                    lines.append(str(doc))
                lines.append("")
        else:
            lines.append(str(documents))
        lines.append("")
    
    # Retry count
    if "retry_count" in answer_data:
        lines.append("## Retry Count")
        lines.append(str(answer_data["retry_count"]))
        lines.append("")
    
    # Tool calls
    if "tool_calls" in answer_data:
        lines.append("## Tool Calls")
        tool_calls = answer_data["tool_calls"]
        if isinstance(tool_calls, list):
            for i, call in enumerate(tool_calls, 1):
                lines.append(f"### Tool Call {i}")
                if isinstance(call, dict):
                    lines.append(f"- **Tool:** {call.get('tool', 'Unknown')}")
                    if 'input' in call:
                        lines.append(f"- **Input:** {json.dumps(call.get('input'), indent=2)}")
                    if 'output' in call:
                        lines.append(f"- **Output:** {json.dumps(call.get('output'), indent=2)}")
                else:
                    lines.append(f"- **Tool:** {str(call)}")
                lines.append("")
        else:
            lines.append(str(tool_calls))
        lines.append("")
    
    return "\n".join(lines)

# --- Main code ---
response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    print(data)
    
    # Convert JSON into a clean report
    content = format_graph_output(data)
    
    # Create folder if it doesn't exist
    folder = "responses"
    os.makedirs(folder, exist_ok=True)
    
    # Filename with timestamp
    now = datetime.datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".md"
    filepath = os.path.join(folder, filename)
    
    # Markdown file content
    md_content = (
        "# API Response Report\n"
        + "="*50 + "\n"
        + f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        + f"**Query:** {payload['query']}\n"
        + "="*50 + "\n\n"
        + content
    )
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"Response saved to {filepath}")
else:
    print("Error:", response.status_code, response.text)