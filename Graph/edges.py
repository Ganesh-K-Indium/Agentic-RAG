"his modules has all info about the graph edges"
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from load_vector_dbs.prompts_and_chains import (get_question_router_chain,
                                                          get_hallucination_chain, 
                                                          get_answer_quality_chain)
from load_vector_dbs.load_dbs import load_vector_database

def route_question(state):
    """
    Route question using integrated approach - prefer vectorstore but prepare for web integration.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION WITH INTEGRATED APPROACH---")
    messages = state["messages"]
    question = messages[-1].content
    init = load_vector_database()
    _, vectorstore, _ = init.get_text_retriever()

    # Generate embedding for the question
    query_embedding = init.embeddings.embed_query(question)

    # Qdrant similarity search for top 5 text docs
    text_results = init.qdrant_client.query_points(
        collection_name=vectorstore.collection_name,
        query=query_embedding,
        limit=5,
        with_payload=True
    )

    # Get image vector store
    image_vectorstore, _, _ = init.get_image_retriever()
    image_results = init.qdrant_client.query_points(
        collection_name=image_vectorstore.collection_name,
        query=query_embedding,
        limit=5,
        with_payload=True
    )

    # Calculate relevance scores
    text_relevance_score = 0
    image_relevance_score = 0
    
    # Extract results safely and calculate scores
    text_points = []
    if hasattr(text_results, 'result'):
        text_points = text_results.result
        text_relevance_score = sum(point.score for point in text_points if hasattr(point, 'score'))
    
    image_points = []
    if hasattr(image_results, 'result'):
        image_points = image_results.result
        image_relevance_score = sum(point.score for point in image_points if hasattr(point, 'score'))

    total_relevance = text_relevance_score + image_relevance_score
    vectorstore_confidence = len(text_points) + len(image_points)

    print(f"Vectorstore relevance score: {total_relevance}, confidence: {vectorstore_confidence}")

    # Enhanced routing logic
    if vectorstore_confidence >= 3 and total_relevance > 0.7:
        print("---ROUTE: HIGH CONFIDENCE VECTORSTORE SEARCH---")
        return "vectorstore"
    elif vectorstore_confidence >= 1 and total_relevance > 0.5:
        print("---ROUTE: MODERATE CONFIDENCE VECTORSTORE SEARCH (may need web supplement)---")
        return "vectorstore"
    else:
        # Check if it's clearly a web-only query
        web_indicators = ["current", "latest", "recent", "news", "today", "this year", "2024", "2025"]
        if any(indicator in question.lower() for indicator in web_indicators):
            print("---ROUTE: WEB SEARCH (temporal/current info needed)---")
            return "web_search"
        else:
            print("---ROUTE: TRY VECTORSTORE FIRST (low confidence, may need web integration)---")
            return "vectorstore"
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, add web search, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    vectorstore_searched = state.get("vectorstore_searched", False)
    web_searched = state.get("web_searched", False)

    if not filtered_documents:
        # All documents have been filtered check_relevance
        if not web_searched and vectorstore_searched:
            print("---DECISION: NO RELEVANT VECTORSTORE DOCS, TRY WEB SEARCH---")
            return "integrate_web_search"
        else:
            print("---DECISION: NO RELEVANT DOCS AFTER ALL SEARCHES, FALLBACK TO FINANCIAL WEB SEARCH---")
            return "financial_web_search"
    else:
        # We have relevant documents, check if we need web supplement
        if vectorstore_searched and not web_searched and len(filtered_documents) < 3:
            print("---DECISION: LIMITED VECTORSTORE RESULTS, SUPPLEMENT WITH WEB SEARCH---")
            return "integrate_web_search"
        else:
            print("---DECISION: SUFFICIENT DOCUMENTS, GENERATE ANSWER---")
            return "generate"
    
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    Enhanced for cross-referencing scenarios.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    messages = state["messages"]
    question = messages[-1].content
    
    documents = state["documents"]
    document_sources = state.get("document_sources", {})
    Intermediate_message = state["Intermediate_message"]
    cross_ref_analysis = state.get("cross_reference_analysis", {})

    #  Track retry count in state
    retry_count = state.get("retry_count", 0)
    max_retries = 2  # or 3, depending on how strict you want to be

    llm = ChatOpenAI(model="gpt-4o")
    hallucination_grader = get_hallucination_chain(llm)
    
    # For cross-referencing, use document_sources if available, otherwise fall back to documents
    docs_for_grading = documents
    if cross_ref_analysis.get("needs_cross_reference") == "yes" and document_sources:
        # Combine all document sources for grading
        all_docs_for_grading = []
        for source_type, source_docs in document_sources.items():
            if source_docs:
                all_docs_for_grading.extend(source_docs)
        if all_docs_for_grading:
            docs_for_grading = all_docs_for_grading
    
    score = hallucination_grader.invoke(
        {"documents": docs_for_grading, "generation": Intermediate_message}
    )
    grade = score.binary_score

    # Check hallucination
    if grade.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        print(f"Question: {question}, Answer {Intermediate_message}")
        answer_grader = get_answer_quality_chain(llm)
        answer_score = answer_grader.invoke(
            {"question": question, "generation": Intermediate_message}
        )
        answer_grade = answer_score.binary_score
        if answer_grade.lower() == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        # For cross-referencing scenarios, be more lenient on retries
        is_cross_ref = cross_ref_analysis.get("needs_cross_reference") == "yes"
        adjusted_max_retries = max_retries + 1 if is_cross_ref else max_retries
        
        if retry_count >= adjusted_max_retries:
            print(f"---MAX RETRIES REACHED ({retry_count}), stopping loop---")
            return "useful"  # fallback → treat as final result instead of looping forever
        else:
            print(f"---DECISION: GENERATION IS NOT GROUNDED, RETRY {retry_count + 1}/{adjusted_max_retries}---")
            # Increment retry count in state
            state["retry_count"] = retry_count + 1
            return "not supported"


def decide_after_web_integration(state):
    """
    Decides next step after web search integration.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        str: Next node to call
    """
    print("---DECIDE AFTER WEB INTEGRATION---")
    documents = state.get("documents", [])
    
    if documents:
        print("---DECISION: DOCUMENTS AVAILABLE AFTER WEB INTEGRATION, GRADE THEM---")
        return "grade_documents"
    else:
        print("---DECISION: NO DOCUMENTS AFTER WEB INTEGRATION, FALLBACK TO FINANCIAL WEB SEARCH---")
        return "financial_web_search"


def decide_cross_reference_approach(state):
    """
    Decides whether to use cross-referencing and citation approach based on question complexity.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        str: Next node to call
    """
    print("---DECIDE CROSS-REFERENCE APPROACH---")
    documents = state.get("documents", [])
    
    # Check if we have multiple document types that might benefit from cross-referencing
    if len(documents) >= 2:
        print("---DECISION: MULTIPLE DOCUMENTS AVAILABLE, ANALYZE CROSS-REFERENCE NEEDS---")
        return "analyze_cross_reference"
    else:
        print("---DECISION: LIMITED DOCUMENTS, USE STANDARD GENERATION---")
        return "generate"


def decide_after_cross_reference_analysis(state):
    """
    Decides next step after cross-reference analysis.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        str: Next node to call
    """
    print("---DECIDE AFTER CROSS-REFERENCE ANALYSIS---")
    cross_ref_analysis = state.get("cross_reference_analysis", {})
    
    if cross_ref_analysis.get("needs_cross_reference") == "yes":
        print("---DECISION: CROSS-REFERENCING NEEDED, CATEGORIZE DOCUMENTS AND DETERMINE STRATEGY---")
        return "categorize_documents"
    else:
        print("---DECISION: NO CROSS-REFERENCING NEEDED, USE STANDARD GENERATION---")
        return "generate"

