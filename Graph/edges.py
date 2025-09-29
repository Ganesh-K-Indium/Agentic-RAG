"his modules has all info about the graph edges"
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from load_vector_dbs.prompts_and_chains import (get_question_router_chain,
                                                          get_hallucination_chain, 
                                                          get_answer_quality_chain)
from load_vector_dbs.load_dbs import load_vector_database

def route_question(state):
    """
    Route question using integrated approach with memory-enhanced routing.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    from Graph.memory_manager import global_memory_manager
    import time

    print("---ROUTE QUESTION WITH INTEGRATED APPROACH---")
    start_time = time.time()
    
    # Initialize memory components
    state = global_memory_manager.initialize_state_memory(state)
    
    messages = state["messages"]
    question = messages[-1].content
    print(f"Question to route: {question}")
    
    # Initialize variables at the top to prevent UnboundLocalError
    text_results = None
    image_results = None
    text_points = []
    image_points = []
    text_relevance_score = 0
    image_relevance_score = 0
    max_text_score = 0
    max_image_score = 0
    
    # Check for routing recommendation from learned patterns
    routing_recommendation = global_memory_manager.get_routing_recommendation(question)
    if routing_recommendation:
        print(f"---MEMORY RECOMMENDED ROUTE: {routing_recommendation}---")
        # Still verify with vectorstore scores but bias towards recommendation
    
    # Check conversation context for better routing
    conversation_context = global_memory_manager.get_conversation_context()
    if conversation_context:
        recent_queries = [ctx['query'] for ctx in conversation_context[-2:]]
        print(f"Recent conversation context: {recent_queries}")
    
    try:
        init = load_vector_database()
        _, vectorstore, _ = init.get_text_retriever()

        # Generate embedding for the question
        query_embedding = init.embeddings.embed_query(question)
        print(f"Generated embedding with dimension: {len(query_embedding)}")
        
        # Check for cached documents first
        cached_docs = global_memory_manager.get_cached_documents(query_embedding)
        if cached_docs:
            print("---USING CACHED DOCUMENT RESULTS---")
            text_points = cached_docs.get('documents', [])
            text_relevance_score = sum(cached_docs.get('scores', [])) / len(cached_docs.get('scores', [1]))
            max_text_score = max(cached_docs.get('scores', [0]))
            image_points = []  # Simplified for cached case
            image_relevance_score = 0
            max_image_score = 0
            state['performance_metrics']['cache_hits'] += 1
        else:
            # Qdrant similarity search for top 5 text docs
            text_results = init.qdrant_client.query_points(
                collection_name=vectorstore.collection_name,
                query=query_embedding,
                limit=5,
                with_payload=True
            )
            print(f"Text search completed for collection: {vectorstore.collection_name}")

            # Get image vector store
            image_vectorstore, _, _ = init.get_image_retriever()
            image_results = init.qdrant_client.query_points(
                collection_name=image_vectorstore.collection_name,
                query=query_embedding,
                limit=5,
                with_payload=True
            )
            print(f"Image search completed for collection: {image_vectorstore.collection_name}")
            
            # Extract results safely and calculate scores
            text_relevance_score = 0
            image_relevance_score = 0
            max_text_score = 0
            max_image_score = 0
            
            text_points = []
            # Qdrant query_points returns points attribute, not result
            if hasattr(text_results, 'points') and text_results.points:
                text_points = text_results.points
                scores = [point.score for point in text_points if hasattr(point, 'score') and point.score is not None]
                if scores:
                    text_relevance_score = sum(scores) / len(scores)  # Average instead of sum
                    max_text_score = max(scores)
                    # Cache the results
                    global_memory_manager.cache_document_retrieval(
                        query_embedding, text_points, scores, "text"
                    )
            elif hasattr(text_results, 'result') and text_results.result:
                # Fallback for older API
                text_points = text_results.result
                scores = [point.score for point in text_points if hasattr(point, 'score') and point.score is not None]
                if scores:
                    text_relevance_score = sum(scores) / len(scores)
                    max_text_score = max(scores)
            
            image_points = []
            # Qdrant query_points returns points attribute, not result
            if hasattr(image_results, 'points') and image_results.points:
                image_points = image_results.points
                scores = [point.score for point in image_points if hasattr(point, 'score') and point.score is not None]
                if scores:
                    image_relevance_score = sum(scores) / len(scores)  # Average instead of sum
                    max_image_score = max(scores)
            elif hasattr(image_results, 'result') and image_results.result:
                # Fallback for older API
                image_points = image_results.result
                scores = [point.score for point in image_points if hasattr(point, 'score') and point.score is not None]
                if scores:
                    image_relevance_score = sum(scores) / len(scores)
                    max_image_score = max(scores)
        
    except Exception as e:
        print(f"Error during vector search: {str(e)}")
        print("---ROUTE: FALLBACK TO VECTORSTORE DUE TO SEARCH ERROR---")
        # Variables are already initialized at the top, so just continue with routing logic

    # Calculate relevance scores more robustly
    text_relevance_score = 0
    image_relevance_score = 0
    max_text_score = 0
    max_image_score = 0
    
    # Extract results safely and calculate scores
    text_points = []
    # Qdrant query_points returns points attribute, not result
    if text_results and hasattr(text_results, 'points') and text_results.points:
        text_points = text_results.points
        scores = [point.score for point in text_points if hasattr(point, 'score') and point.score is not None]
        if scores:
            text_relevance_score = sum(scores) / len(scores)  # Average instead of sum
            max_text_score = max(scores)
    elif hasattr(text_results, 'result') and text_results.result:
        # Fallback for older API
        text_points = text_results.result
        scores = [point.score for point in text_points if hasattr(point, 'score') and point.score is not None]
        if scores:
            text_relevance_score = sum(scores) / len(scores)
            max_text_score = max(scores)
    
    image_points = []
    # Qdrant query_points returns points attribute, not result
    if image_results and hasattr(image_results, 'points') and image_results.points:
        image_points = image_results.points
        scores = [point.score for point in image_points if hasattr(point, 'score') and point.score is not None]
        if scores:
            image_relevance_score = sum(scores) / len(scores)  # Average instead of sum
            max_image_score = max(scores)
    elif hasattr(image_results, 'result') and image_results.result:
        # Fallback for older API
        image_points = image_results.result
        scores = [point.score for point in image_points if hasattr(point, 'score') and point.score is not None]
        if scores:
            image_relevance_score = sum(scores) / len(scores)
            max_image_score = max(scores)

    # Use the higher of average scores or max individual score for decision making
    best_text_score = max(text_relevance_score, max_text_score)
    best_image_score = max(image_relevance_score, max_image_score)
    best_overall_score = max(best_text_score, best_image_score)
    average_relevance = (text_relevance_score + image_relevance_score) / 2 if (text_relevance_score > 0 or image_relevance_score > 0) else 0
    
    vectorstore_confidence = len(text_points) + len(image_points)

    print(f"Text: avg={text_relevance_score:.3f}, max={max_text_score:.3f}, docs={len(text_points)}")
    print(f"Image: avg={image_relevance_score:.3f}, max={max_image_score:.3f}, docs={len(image_points)}")
    print(f"Best overall score: {best_overall_score:.3f}, Average: {average_relevance:.3f}, Total docs: {vectorstore_confidence}")

    # Enhanced routing logic with memory-informed decisions
    # Check if it's clearly a web-only query first with stronger temporal detection
    strong_temporal_indicators = ["today", "current", "latest", "recent", "news", "this year", "now"]
    stock_price_indicators = ["stock price", "market cap", "share price", "current price"]
    temporal_years = ["2024", "2025", "this month", "this week"]
    
    # Strong temporal query detection
    has_strong_temporal = any(indicator in question.lower() for indicator in strong_temporal_indicators)
    has_stock_price = any(indicator in question.lower() for indicator in stock_price_indicators)
    has_temporal_year = any(indicator in question.lower() for indicator in temporal_years)
    
    is_strong_temporal_query = has_strong_temporal or has_stock_price or has_temporal_year
    
    # Memory-informed routing decision
    routing_decision = None
    
    # Apply routing recommendation if available and scores support it
    if routing_recommendation:
        if routing_recommendation == "vectorstore" and best_overall_score > 0.3:
            routing_decision = "vectorstore"
            print(f"---ROUTE: MEMORY-OPTIMIZED VECTORSTORE (learned pattern, score: {best_overall_score:.3f})---")
        elif routing_recommendation == "web_search" and is_strong_temporal_query:
            routing_decision = "web_search"
            print("---ROUTE: MEMORY-OPTIMIZED WEB SEARCH (learned pattern for temporal queries)---")
    
    # If no memory recommendation or it doesn't apply, use standard logic
    if not routing_decision:
        # For strong temporal queries, prioritize web search more aggressively
        if is_strong_temporal_query:
            if has_stock_price or "today" in question.lower() or "current" in question.lower():
                routing_decision = "web_search"
                print("---ROUTE: WEB SEARCH (strong temporal/real-time info needed)---")
            elif best_overall_score < 0.6:  # Higher threshold for temporal queries
                routing_decision = "web_search"
                print("---ROUTE: WEB SEARCH (temporal info needed, moderate vectorstore relevance)---")
        
        # Standard routing for non-temporal queries
        if not routing_decision:
            if vectorstore_confidence >= 3 and best_overall_score > 0.4:
                routing_decision = "vectorstore"
                print("---ROUTE: HIGH CONFIDENCE VECTORSTORE SEARCH---")
            elif vectorstore_confidence >= 2 and best_overall_score > 0.3:
                routing_decision = "vectorstore"
                print("---ROUTE: GOOD CONFIDENCE VECTORSTORE SEARCH---")
            elif vectorstore_confidence >= 1 and best_overall_score > 0.2:
                routing_decision = "vectorstore"
                print("---ROUTE: MODERATE CONFIDENCE VECTORSTORE SEARCH (may need web supplement)---")
            elif vectorstore_confidence >= 1:
                # Documents found but low scores - still try vectorstore first for complex queries
                routing_decision = "vectorstore"
                print(f"---ROUTE: VECTORSTORE FIRST (documents found but low scores: {best_overall_score:.3f})---")
            else:
                if is_strong_temporal_query:
                    routing_decision = "web_search"
                    print("---ROUTE: WEB SEARCH (temporal query, no vectorstore docs)---")
                else:
                    routing_decision = "vectorstore"
                    print("---ROUTE: TRY VECTORSTORE FIRST (no docs found, fallback attempt)---")
    
    # Record routing decision for learning (with safety check)
    response_time = time.time() - start_time
    
    # Safety check: ensure routing_decision is never None
    if routing_decision is None:
        routing_decision = "vectorstore"  # Default fallback
        print("---ROUTE: FALLBACK TO VECTORSTORE (no decision made)---")
    
    state['routing_memory'] = {
        'decision': routing_decision,
        'scores': {'text': best_text_score, 'image': best_image_score, 'overall': best_overall_score},
        'response_time': response_time,
        'temporal_query': is_strong_temporal_query,
        'recommendation_used': routing_recommendation is not None
    }
    
    # Update performance metrics
    state['performance_metrics']['total_queries'] += 1
    
    return routing_decision
    
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
    
    print(f"Filtered documents count: {len(filtered_documents) if filtered_documents else 0}")
    print(f"Vectorstore searched: {vectorstore_searched}, Web searched: {web_searched}")

    if not filtered_documents:
        # All documents have been filtered check_relevance
        retry_count = state.get("retry_count", 0)
        if not web_searched and vectorstore_searched and retry_count < 2:
            print("---DECISION: NO RELEVANT VECTORSTORE DOCS, TRY WEB SEARCH---")
            return "integrate_web_search"
        elif not web_searched and retry_count < 2:
            print("---DECISION: NO RELEVANT DOCS AFTER ALL SEARCHES, TRY FINANCIAL WEB SEARCH---")
            return "financial_web_search"
        else:
            print("---DECISION: NO RELEVANT DOCS FOUND, GENERATE WITH AVAILABLE CONTEXT---")
            # Force generation even without perfect documents to prevent infinite loops
            return "generate"
    else:
        # We have relevant documents, be more generous with what we consider sufficient
        if vectorstore_searched and not web_searched and len(filtered_documents) == 1:
            # Only supplement with web if we have just 1 document (reduced threshold)
            print("---DECISION: SINGLE DOCUMENT FOUND, SUPPLEMENT WITH WEB SEARCH---")
            return "integrate_web_search"
        else:
            print(f"---DECISION: SUFFICIENT DOCUMENTS ({len(filtered_documents)}), GENERATE ANSWER---")
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
    max_retries = 1  # Reduced from 2 to 1 for faster responses

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
            # Be more lenient to avoid excessive retries
            if retry_count >= 1:  # Accept after 1 retry instead of continuing
                print("---DECISION: ACCEPTING GENERATION TO AVOID EXCESSIVE RETRIES---")
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
            # Force acceptance to prevent infinite loops
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
    Always categorizes documents to generate citations, then determines strategy.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        str: Next node to call
    """
    print("---DECIDE AFTER CROSS-REFERENCE ANALYSIS---")
    cross_ref_analysis = state.get("cross_reference_analysis", {})
    
    # Always categorize documents to generate proper citations and source tracking
    print("---DECISION: CATEGORIZING DOCUMENTS FOR CITATIONS AND SOURCE TRACKING---")
    return "categorize_documents"

