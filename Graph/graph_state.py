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
    