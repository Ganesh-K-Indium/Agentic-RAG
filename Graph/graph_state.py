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
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    Intermediate_message: str
    documents: List[str]
    retry_count: int
    tool_calls: List[Dict[str, Any]] 
    