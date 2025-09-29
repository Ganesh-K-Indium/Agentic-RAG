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

def _extract_document_source(metadata: dict) -> str:
    """
    Extract meaningful source information from document metadata.
    
    Args:
        metadata: Document metadata dictionary
        
    Returns:
        str: Formatted source string with company, file, and page info
    """
    if not metadata:
        return "Unknown"
    
    # Try to build a meaningful source string from available metadata
    source_parts = []
    
    # Add company information if available
    if metadata.get('company'):
        company = metadata['company'].upper()
        source_parts.append(company)
    
    # Add source file information
    if metadata.get('source_file'):
        source_file = metadata['source_file']
        # Remove file extension for cleaner display
        if '.' in source_file:
            source_file = source_file.rsplit('.', 1)[0]
        source_parts.append(source_file)
    
    # Add page number if available
    if metadata.get('page_num') is not None:
        source_parts.append(f"Page {metadata['page_num']}")
    
    # Add content type if available and useful
    if metadata.get('content_type') and metadata['content_type'] != 'text':
        source_parts.append(f"({metadata['content_type']})")
    
    # If we have web search results, handle differently
    if 'url' in metadata:
        url = metadata['url']
        # Extract domain name for cleaner display
        if url.startswith('http'):
            domain = url.split('/')[2] if len(url.split('/')) > 2 else url
            source_parts.append(f"Web: {domain}")
    
    # Fallback to basic metadata keys
    if not source_parts:
        if metadata.get('source'):
            return str(metadata['source'])
        elif metadata.get('_collection_name'):
            return f"Collection: {metadata['_collection_name']}"
        else:
            return "Document"
    
    return " - ".join(source_parts)

# Input schema
class QueryInput(BaseModel):
    query: str
    user_id: str = "anonymous"  # Default to anonymous if not provided
    extra_inputs: dict = None

@app.post("/ask")
async def ask_agent(payload: QueryInput):
    """Route queries to ManagerAgent with session awareness."""
    query = payload.query
    user_id = payload.user_id
    extra_inputs = payload.extra_inputs
    
    # Handle with session awareness
    result = manager.handle(query, user_id=user_id, extra_inputs=extra_inputs)
    print(f"Query from user {user_id}: {query}")
    print(f"Session info: {result.get('session_info', {})}")
    
    # Prepare response with session information
    routing_decision = 'unknown'
    if result.get('routing_memory') and result.get('routing_memory', {}).get('decision'):
        routing_decision = result['routing_memory']['decision']
    elif result.get('vectorstore_searched') and result.get('web_searched'):
        routing_decision = 'vectorstore_with_web_supplement'
    elif result.get('vectorstore_searched'):
        routing_decision = 'vectorstore'
    elif result.get('web_searched'):
        routing_decision = 'web_search'
    
    # Extract comprehensive response data
    documents = result.get('documents', [])
    citation_info = result.get('citation_info', [])
    tool_calls = result.get('tool_calls', [])
    document_sources = result.get('document_sources', {})
    cross_reference_analysis = result.get('cross_reference_analysis', {})
    
    response_data = {
        "answer": result.get('Intermediate_message', result.get('messages', [{}])[-1]),
        "session_info": result.get('session_info', {}),
        "documents_used": len(documents),
        "routing_decision": routing_decision,
        "documents": [
            {
                "content": (doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content) if hasattr(doc, 'page_content') else str(doc)[:500] + "...",
                "metadata": getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else {},
                "source": _extract_document_source(getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else {}),
                "type": "LangChain Document" if hasattr(doc, 'page_content') else "Other"
            } for doc in documents
        ],
        "citations": citation_info,
        "tool_calls": tool_calls,
        "document_sources": {
            k: len(v) for k, v in document_sources.items() if v
        },
        "cross_reference_analysis": cross_reference_analysis,
        "performance_metrics": {
            "vectorstore_searched": result.get('vectorstore_searched', False),
            "web_searched": result.get('web_searched', False),
            "vectorstore_quality": result.get('vectorstore_quality', 'none'),
            "retry_count": result.get('retry_count', 0)
        }
    }
    
    # Log the response to markdown file
    log_response(payload.dict(), response_data)
    
    return response_data

# Additional endpoints for session management
@app.get("/session/{user_id}")
async def get_user_session(user_id: str):
    """Get session summary for a specific user."""
    session_summary = manager.get_user_session_summary(user_id)
    return {"user_id": user_id, "session_summary": session_summary}

@app.delete("/session/{user_id}")
async def clear_user_session(user_id: str):
    """Clear/reset a user's session."""
    success = manager.clear_user_session(user_id)
    return {"user_id": user_id, "session_cleared": success}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Agentic RAG API"}