import os
from fastapi import FastAPI
from pydantic import BaseModel
from manager_agent.manager import ManagerAgent  #  import your manager

# Initialize FastAPI + ManagerAgent
app = FastAPI()
manager = ManagerAgent()

# Input schema
class QueryInput(BaseModel):
    query: str

@app.post("/ask")
async def ask_agent(payload: QueryInput):
    """Route queries to ManagerAgent (RAG or Ingestion)."""
    query = payload.query
    result = manager.handle(query)
    return {"answer": result}
