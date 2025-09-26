"This module contains all info about about the nodes in the graph"
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_tavily import TavilySearch
from load_vector_dbs.prompts_and_chains import (get_retrival_grader_chain, get_rag_chain,
                                                          get_company_name, get_question_rewriter_chain,
                                                          get_cross_reference_analyzer_chain,
                                                          get_document_summary_strategy_chain,
                                                          get_enhanced_rag_chain_with_citations)
from load_vector_dbs.load_dbs import load_vector_database
load_dotenv()

def extract_multiple_companies_from_question(question, llm=None):
    """
    Extract multiple companies from the question for cross-referencing scenarios.
    Uses LLM-based extraction for more accurate company identification.
    """
    try:
        if llm:
            from load_vector_dbs.prompts_and_chains import get_multi_company_extractor_chain
            
            # Use structured LLM extraction
            extractor_chain = get_multi_company_extractor_chain(llm)
            extraction_result = extractor_chain.invoke({"question": question})
            
            print(f"Extracted companies: {extraction_result.companies}, Primary: {extraction_result.primary_company}, Is comparison: {extraction_result.is_comparison}")
            
            return extraction_result.companies
        
    except Exception as e:
        print(f"Error in LLM-based company extraction: {e}")
    
    # Fallback to keyword matching
    company_mappings = {
        "amazon": ["amazon", "amzn", "amazon.com"],
        "berkshire": ["berkshire", "berkshire hathaway", "brk"],
        "google": ["google", "alphabet", "googl", "goog"],
        "Jhonson and Jhonosn": ["johnson", "jnj", "johnson & johnson", "johnson and johnson"],
        "jp morgan": ["jp morgan", "jpmorgan", "jpmc", "chase", "jpm"],
        "meta": ["meta", "facebook", "fb", "meta platforms"],
        "microsoft": ["microsoft", "msft", "ms"],
        "nvidia": ["nvidia", "nvda"],
        "tesla": ["tesla", "tsla"],
        "visa": ["visa", "v"],
        "walmart": ["walmart", "wmt"],
        "pfizer": ["pfizer", "pfe"]
    }
    
    question_lower = question.lower()
    detected_companies = []
    
    for standard_name, variations in company_mappings.items():
        for variation in variations:
            if variation in question_lower:
                if standard_name not in detected_companies:
                    detected_companies.append(standard_name)
                break
    
    # Also look for comparison keywords to determine if cross-referencing is likely
    comparison_keywords = ["compare", "versus", "vs", "against", "with", "between", "and"]
    has_comparison = any(keyword in question_lower for keyword in comparison_keywords)
    
    # If we found multiple companies or comparison keywords, return detected companies
    if len(detected_companies) > 1 or (len(detected_companies) >= 1 and has_comparison):
        return detected_companies
    
    # If only one company detected but no comparison context, still return it
    return detected_companies
def retrieve(state):
    print("---RETRIEVE---")
    messages = state["messages"]
    question = messages[-1].content

    init = load_vector_database()
    retriever, _, _ = init.get_text_retriever()
    documents = retriever.invoke(question)

    tool_call_entry = {
        "tool": "text_retriever"
    }

    print(f"DOCUMENTS RETRIEVED AND NUMBER OF DOCUMENTS ARE {len(documents)}")
    return {
        "documents": documents,
        "vectorstore_searched": True,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def retrieve_from_images_data(state):
    print("---RETRIEVE FROM IMAGES---")
    messages = state["messages"]
    question = messages[-1].content
    documents = state["documents"]

    init = load_vector_database()
    _, image_retriever, _ = init.get_image_retriever()
    results = image_retriever.invoke(question)

    # Check if we need cross-referencing (multiple companies)
    cross_ref_analysis = state.get("cross_reference_analysis", {})
    needs_cross_reference = cross_ref_analysis.get("needs_cross_reference", "no")
    
    if needs_cross_reference == "yes":
        print("---CROSS-REFERENCING MODE: RETRIEVING IMAGES FROM MULTIPLE SOURCES---")
        # For cross-referencing, we want images from all relevant companies
        # Extract multiple companies from the question
        llm = ChatOpenAI(model="gpt-4o-mini")
        companies_in_question = extract_multiple_companies_from_question(question, llm)
        
        if companies_in_question:
            print(f"DETECTED COMPANIES FOR CROSS-REFERENCING: {companies_in_question}")
            # Filter results to include images from all detected companies
            filtered_results = [
                doc for doc in results
                if any(company.lower() in doc.metadata.get("company", "").lower() 
                      for company in companies_in_question)
            ]
        else:
            # If no specific companies detected, use top relevant images regardless of company
            print("NO SPECIFIC COMPANIES DETECTED, USING TOP RELEVANT IMAGES")
            filtered_results = results[:6]  # Get more images for cross-referencing
    else:
        print("---SINGLE COMPANY MODE: FILTERING TO PRIMARY COMPANY---")
        # Original single-company filtering logic
        llm = ChatOpenAI(model="gpt-4o-mini")
        company_extractor = get_company_name(llm)
        company = company_extractor.invoke({"question": question})
        print(f"PRIMARY COMPANY: {company}")

        filtered_results = [
            doc for doc in results
            if doc.metadata.get("company", "").lower() in company.company.lower()
        ]

    if filtered_results:
        documents.extend(filtered_results)

    tool_call_entry = {
        "tool": "image_retriever"
    }

    print(f"UPDATED DOCUMENTS LENGTH: {len(documents)} (Added {len(filtered_results)} images)")
    return {
        "documents": documents,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[-1].content
    documents = state["documents"]
    
    # Check if cross-referencing analysis has been done
    cross_ref_analysis = state.get("cross_reference_analysis", {})
    
    if cross_ref_analysis.get("needs_cross_reference") == "yes":
        print("---USING ENHANCED GENERATION WITH CROSS-REFERENCING---")
        # Use the enhanced generation with cross-referencing
        return generate_with_cross_reference_and_citations(state)
    else:
        print("---USING STANDARD GENERATION---")
        # Use standard generation for simple queries
        llm = ChatOpenAI(model="gpt-4o-mini")
        rag_chain = get_rag_chain(llm)
        Intermediate_message = rag_chain.invoke(
            {"documents": documents, "question": question}
        )

        retry_count = state.get("retry_count", 0)

        tool_call_entry = {
            "tool": "rag_chain"
        }

        return {
            "Intermediate_message": Intermediate_message,
            "retry_count": retry_count + 1,
            "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
        }


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE---")
    messages = state["messages"]
    question = messages[-1].content
    documents = state["documents"]
    cross_ref_analysis = state.get("cross_reference_analysis", {})

    filtered_docs = []
    llm_grade_document = ChatOpenAI(model="gpt-4o")
    retrieval_grader = get_retrival_grader_chain(llm_grade_document)

    results_log = []
    is_cross_ref = cross_ref_analysis.get("needs_cross_reference") == "yes"
    
    # For cross-referencing, be more inclusive with document filtering
    min_docs_threshold = 5 if is_cross_ref else 3
    
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        results_log.append({"doc": d.page_content[:100] + "...", "grade": grade})
        if grade.lower() == "yes":
            filtered_docs.append(d)

    # If we're doing cross-referencing and have too few docs, be more lenient
    if is_cross_ref and len(filtered_docs) < min_docs_threshold:
        print(f"---CROSS-REFERENCING MODE: Only {len(filtered_docs)} docs passed grading, being more inclusive---")
        # Add some documents that were close to passing
        remaining_docs = [d for d in documents if d not in filtered_docs]
        if remaining_docs:
            # Add up to the difference needed to reach threshold
            additional_needed = min(min_docs_threshold - len(filtered_docs), len(remaining_docs))
            filtered_docs.extend(remaining_docs[:additional_needed])
            print(f"---ADDED {additional_needed} ADDITIONAL DOCS FOR CROSS-REFERENCING---")

    tool_call_entry = {
        "tool": "retrieval_grader"
    }

    print(f"FILTERED DOCS COUNT: {len(filtered_docs)} (Cross-ref mode: {is_cross_ref})")
    return {
        "documents": filtered_docs,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def transform_query(state):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[-1].content

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    question_rewriter = get_question_rewriter_chain(llm)
    better_question = question_rewriter.invoke({"question": question})

    tool_call_entry = {
        "tool": "question_rewriter"
    }

    return {
        "messages": [better_question],
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def web_search(state):
    print("---WEB SEARCH---")
    messages = state["messages"]
    question = messages[-1].content
    web_search_tool = TavilySearch(k=3)
    docs = web_search_tool.invoke({"query": question})

    if isinstance(docs, list):
        web_results = "\n".join(
            d["content"] if isinstance(d, dict) and "content" in d else str(d)
            for d in docs
        )
    else:
        web_results = str(docs)

    tool_call_entry = {
        "tool": "web_search"
    }

    return {
        "documents": [Document(page_content=web_results)],
        "web_searched": True,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def financial_web_search(state):
    print("---FINANCIAL WEB SEARCH---")
    messages = state["messages"]
    question = messages[-1].content

    web_search_tool = TavilySearch(k=3)
    docs = web_search_tool.invoke({"query": question})

    if isinstance(docs, list):
        web_results = "\n".join(
            d.get("content", "") if isinstance(d, dict) else str(d)
            for d in docs
        )
    else:
        web_results = str(docs)

    tool_call_entry = {
        "tool": "financial_web_search"
    }

    return {
        "documents": [Document(page_content=web_results)],
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def integrate_web_search(state):
    """
    Perform web search to supplement vectorstore results when they are insufficient.
    """
    print("---INTEGRATE WEB SEARCH---")
    messages = state["messages"]
    question = messages[-1].content
    existing_documents = state.get("documents", [])

    web_search_tool = TavilySearch(k=3)
    docs = web_search_tool.invoke({"query": question})

    if isinstance(docs, list):
        web_results = "\n".join(
            d.get("content", "") if isinstance(d, dict) else str(d)
            for d in docs
        )
    else:
        web_results = str(docs)

    # Add web results to existing documents
    web_doc = Document(page_content=web_results)
    combined_documents = existing_documents + [web_doc]

    tool_call_entry = {
        "tool": "integrate_web_search"
    }

    print(f"INTEGRATED WEB SEARCH RESULTS WITH {len(existing_documents)} EXISTING DOCS")
    return {
        "documents": combined_documents,
        "web_searched": True,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def evaluate_vectorstore_quality(state):
    """
    Evaluate the quality of vectorstore results to determine if web search is needed.
    """
    print("---EVALUATE VECTORSTORE QUALITY---")
    messages = state["messages"]
    question = messages[-1].content
    documents = state.get("documents", [])

    # Simple heuristics to evaluate vectorstore quality
    quality = "none"
    needs_web_fallback = True

    if documents:
        # Check if we have sufficient relevant documents
        if len(documents) >= 2:
            quality = "good"
            needs_web_fallback = False
        elif len(documents) == 1:
            quality = "poor"
            needs_web_fallback = True
        else:
            quality = "none"
            needs_web_fallback = True
    
    tool_call_entry = {
        "tool": "evaluate_vectorstore_quality"
    }

    print(f"VECTORSTORE QUALITY: {quality}, NEEDS WEB FALLBACK: {needs_web_fallback}")
    return {
        "vectorstore_quality": quality,
        "needs_web_fallback": needs_web_fallback,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def analyze_cross_reference_needs(state):
    """
    Analyze if the question requires cross-referencing across multiple sources.
    """
    print("---ANALYZE CROSS-REFERENCE NEEDS---")
    messages = state["messages"]
    question = messages[-1].content

    llm = ChatOpenAI(model="gpt-4o")
    cross_ref_analyzer = get_cross_reference_analyzer_chain(llm)
    
    analysis = cross_ref_analyzer.invoke({"question": question})
    
    tool_call_entry = {
        "tool": "cross_reference_analyzer"
    }

    print(f"CROSS-REFERENCE ANALYSIS: {analysis}")
    return {
        "cross_reference_analysis": {
            "needs_cross_reference": analysis.needs_cross_reference,
            "source_types_needed": analysis.source_types_needed,
            "reasoning": analysis.reasoning
        },
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def determine_summary_strategy(state):
    """
    Determine the optimal strategy for document summarization based on cross-reference analysis.
    """
    print("---DETERMINE SUMMARY STRATEGY---")
    messages = state["messages"]
    question = messages[-1].content
    cross_ref_analysis = state.get("cross_reference_analysis", {})
    
    # Determine available sources based on current state
    available_sources = []
    if state.get("vectorstore_searched", False):
        available_sources.extend(["text_docs", "images"])
    if state.get("web_searched", False):
        available_sources.append("web_search")
        
    llm = ChatOpenAI(model="gpt-4o")
    strategy_analyzer = get_document_summary_strategy_chain(llm)
    
    strategy = strategy_analyzer.invoke({
        "question": question,
        "document_sources": available_sources
    })
    
    tool_call_entry = {
        "tool": "summary_strategy_analyzer"
    }

    print(f"SUMMARY STRATEGY: {strategy.strategy}")
    return {
        "summary_strategy": strategy.strategy,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def categorize_documents_by_source(state):
    """
    Categorize documents by their source type for proper citation and cross-referencing.
    """
    print("---CATEGORIZE DOCUMENTS BY SOURCE---")
    documents = state.get("documents", [])
    
    document_sources = {
        "vectorstore_text": [],
        "vectorstore_images": [],
        "web_search": [],
        "financial_web": []
    }
    
    citation_info = []
    
    for idx, doc in enumerate(documents):
        # Determine source type based on document metadata or content patterns
        source_type = "vectorstore_text"  # default
        
        if hasattr(doc, 'metadata'):
            metadata = doc.metadata
            # Check if it's an image document
            if "This is an image with the caption:" in doc.page_content:
                source_type = "vectorstore_images"
            # Check if it's from web search (added by integrate_web_search or web_search nodes)
            elif any(indicator in str(metadata).lower() for indicator in ["tavily", "web", "search"]):
                source_type = "web_search"
            # Check if it's from financial web search
            elif "financial" in str(metadata).lower():
                source_type = "financial_web"
        else:
            # Fallback: analyze content for source type hints
            if "This is an image with the caption:" in doc.page_content:
                source_type = "vectorstore_images"
            elif any(indicator in doc.page_content.lower()[:200] for indicator in ["breaking news", "current", "today", "latest"]):
                source_type = "web_search"
        
        document_sources[source_type].append(doc)
        
        # Create citation info
        citation_info.append({
            "source_type": source_type,
            "document_id": f"doc_{idx}",
            "relevance_score": 0.8,  # Would be better to get actual score
            "key_information": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        })
    
    tool_call_entry = {
        "tool": "document_categorizer"
    }

    print(f"DOCUMENT CATEGORIZATION: {[(k, len(v)) for k, v in document_sources.items()]}")
    return {
        "document_sources": document_sources,
        "citation_info": citation_info,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def generate_with_cross_reference_and_citations(state):
    """
    Generate response using cross-referencing and proper citations.
    """
    print("---GENERATE WITH CROSS-REFERENCE AND CITATIONS---")
    messages = state["messages"]
    question = messages[-1].content
    document_sources = state.get("document_sources", {})
    summary_strategy = state.get("summary_strategy", "single_source")
    cross_reference_analysis = state.get("cross_reference_analysis", {})

    llm = ChatOpenAI(model="gpt-4o")
    enhanced_rag_chain = get_enhanced_rag_chain_with_citations(llm)
    
    # Format document sources for the prompt
    formatted_sources = {}
    for source_type, docs in document_sources.items():
        if docs:
            formatted_sources[source_type] = [
                {"content": doc.page_content, "metadata": getattr(doc, 'metadata', {})}
                for doc in docs
            ]
    
    enhanced_response = enhanced_rag_chain.invoke({
        "question": question,
        "document_sources": formatted_sources,
        "summary_strategy": summary_strategy,
        "cross_reference_analysis": cross_reference_analysis
    })

    retry_count = state.get("retry_count", 0)

    tool_call_entry = {
        "tool": "enhanced_rag_chain_with_citations"
    }

    print("GENERATED ENHANCED RESPONSE WITH CITATIONS")
    return {
        "Intermediate_message": enhanced_response,
        "retry_count": retry_count + 1,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


def show_result(state):
    print("---SHOW RESULT---")
    Final_answer = AIMessage(content=state["Intermediate_message"])

    tool_call_entry = {
        "tool": "final_output"
    }

    print(f'SHOWING THE RESULTS: {Final_answer}')
    return {
        "messages": Final_answer,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }
