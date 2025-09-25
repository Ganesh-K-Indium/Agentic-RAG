import os
from fastapi import FastAPI
from pydantic import BaseModel
from manager_agent.manager import ManagerAgent  #  import your manager
from app_logger import log_response
# Initialize FastAPI + ManagerAgent
app = FastAPI()
manager = ManagerAgent()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class QueryInput(BaseModel):
    query: str

@app.post("/ask")
async def ask_agent(payload: QueryInput):
    """Route queries to ManagerAgent (RAG or Ingestion)."""
    query = payload.query
    result = manager.handle(query)
    print(result)
    
    # Log the response to markdown file
    response_data = {"answer": result}
    log_response(payload.dict(), response_data)
    
    return response_data