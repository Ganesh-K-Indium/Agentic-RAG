from typing import List, Annotated, Sequence,Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

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
    