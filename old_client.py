import requests
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
# API endpoint
url = "http://127.0.0.1:8000/ask"  

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Example query
# payload = {
#     "query": "Get Pdf files from confluence data source with space key test and ingest them"
# }

# payload = {
#     "query": "Explain about the year on year financial performance of META from year 2023 to 2024 use Vector Store and financial web search"
# }
# payload = {
#     "query": "What was PFIZER's revenue in 2022?"    
# }

payload = {
    "query": """
        What do you know about amazon"""
}

# Send request
response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()

    # Convert entire JSON to string (no assumptions about structure)
    raw_text = json.dumps(data, indent=2)

    # Use GPT-4o to beautify response
    llm = ChatOpenAI(model="gpt-4o")

    beautify_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are an assistant that converts raw API JSON responses into a neat, human-friendly CLI report.\n"
     "- If the response contains structured fields (e.g., request, source, file_name, etc.), present them as a human-friendly command line report.\n"
     "- If the response contains logs (which are usually a list of steps), format them as a numbered or bulleted checklist for readability.\n"
     "- Ensure formatting is simple and suitable for plain CLI .\n
     "- Always maximize clarity and readability."""),
        ("human", "Here is the raw JSON or log response:\n\n{raw_text}\n\nFormat and present it cleanly for CLI display.")
    ])

    beautify_chain = beautify_prompt | llm
    pretty_response = beautify_chain.invoke({"raw_text": raw_text})

    print("\n" + "="*50)
    print("Final Response")
    print("="*50)
    print(pretty_response.content)

else:
    print("Error:", response.status_code, response.text)