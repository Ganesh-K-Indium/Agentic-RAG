"This module is useful for pydantic classes"
from typing import Literal
from pydantic import BaseModel, Field

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
    
class ExtractCompany(BaseModel):
    """Route a user query to the most relevant datasource."""

    company: str = Field(
        description="Given a user question extract the company name from the question.",
    )
    
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class CrossReferenceAnalysis(BaseModel):
    """Analysis of whether cross-referencing across multiple sources is needed."""
    
    needs_cross_reference: str = Field(
        description="Whether cross-referencing is needed, 'yes' or 'no'"
    )
    source_types_needed: list[str] = Field(
        description="List of source types needed: 'text_docs', 'images', 'web_search', 'financial_data'"
    )
    reasoning: str = Field(
        description="Brief explanation of why cross-referencing is or isn't needed"
    )

class DocumentSummaryStrategy(BaseModel):
    """Strategy for summarizing documents based on cross-reference analysis."""
    
    strategy: Literal["single_source", "multi_source_vectorstore", "integrated_web_vectorstore"] = Field(
        description="Strategy for document summarization"
    )
    primary_sources: list[str] = Field(
        description="Primary document sources to use"
    )
    supplementary_sources: list[str] = Field(
        description="Supplementary sources if needed"
    )

class CitationInfo(BaseModel):
    """Information for document citation."""
    
    source_type: Literal["vectorstore_text", "vectorstore_image", "web_search", "financial_web"] = Field(
        description="Type of source being cited"
    )
    document_id: str = Field(
        description="Identifier for the document"
    )
    relevance_score: float = Field(
        description="Relevance score of the document to the query"
    )
    key_information: str = Field(
        description="Key information extracted from this source"
    )

class MultiCompanyExtraction(BaseModel):
    """Extract multiple companies from a question for cross-referencing."""
    
    companies: list[str] = Field(
        description="List of company names found in the question"
    )
    primary_company: str = Field(
        description="The primary/main company being discussed"
    )
    is_comparison: bool = Field(
        description="Whether the question involves comparison between companies"
    )