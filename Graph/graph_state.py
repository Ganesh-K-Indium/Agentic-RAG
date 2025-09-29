from typing import List, Annotated, Sequence,Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import time

class GraphState(TypedDict):
    """
    Represents the state of our graph with memory capabilities.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        vectorstore_searched: whether vectorstore has been searched
        web_searched: whether web search has been conducted
        vectorstore_quality: quality score of vectorstore results
        needs_web_fallback: whether web search is needed as fallback
        cross_reference_analysis: analysis of cross-referencing needs
        document_sources: categorized document sources for citation
        citation_info: citation information for all sources
        summary_strategy: strategy for document summarization
        
        # Memory and Performance Enhancement Components
        conversation_memory: conversational context and history
        query_cache: cached results for similar queries
        document_cache: cached document retrievals and gradings
        routing_memory: learned routing patterns and preferences
        performance_metrics: timing and quality metrics for optimization
        context_embeddings: cached embeddings for faster similarity matching
        user_preferences: learned user preferences and query patterns
        session_metadata: session-level information and context
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    Intermediate_message: str
    documents: List[str]
    retry_count: int
    tool_calls: List[Dict[str, Any]]
    vectorstore_searched: bool
    web_searched: bool
    vectorstore_quality: str  # "good", "poor", "none"
    needs_web_fallback: bool
    cross_reference_analysis: Dict[str, Any]
    document_sources: Dict[str, List[Any]]  # categorized by source type
    citation_info: List[Dict[str, Any]]
    summary_strategy: str
    
    # Memory and Performance Enhancement
    conversation_memory: Optional[Dict[str, Any]]
    query_cache: Optional[Dict[str, Any]]  
    document_cache: Optional[Dict[str, Any]]
    routing_memory: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]
    context_embeddings: Optional[Dict[str, Any]]
    user_preferences: Optional[Dict[str, Any]]
    session_metadata: Optional[Dict[str, Any]] 
    