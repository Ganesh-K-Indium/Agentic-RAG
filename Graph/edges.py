"his modules has all info about the graph edges"
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from load_vector_dbs.prompts_and_chains import (get_question_router_chain,
                                                          get_hallucination_chain, 
                                                          get_answer_quality_chain)
from load_vector_dbs.load_dbs import load_vector_database

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
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

    # Extract file names or summaries from text payloads
    top_text_docs = []
    # Debug print the QueryResponse object
    print(f"Text QueryResponse: {text_results}")
    
    # Extract results safely
    text_points = []
    if hasattr(text_results, 'result'):
        text_points = text_results.result
    print(f"Top text results from Qdrant (count: {len(text_points)}):")
    
    for idx, point in enumerate(text_points):
        try:
            if hasattr(point, 'payload'):
                payload = point.payload
                print(f"  [text result {idx}] payload: {payload}")
                doc_name = payload.get("source_file", payload.get("summary", "Unknown"))
                print(f"    extracted: {doc_name}")
                top_text_docs.append(doc_name)
        except Exception as e:
            print(f"Error processing text result {idx}: {e}")
            continue

    top_image_docs = []
    # Debug print the QueryResponse object
    print(f"Image QueryResponse: {image_results}")
    
    # Extract results safely
    image_points = []
    if hasattr(image_results, 'result'):
        image_points = image_results.result
    print(f"Top image results from Qdrant (count: {len(image_points)}):")
    
    for idx, point in enumerate(image_points):
        try:
            if hasattr(point, 'payload'):
                payload = point.payload
                print(f"  [image result {idx}] payload: {payload}")
                img_info = payload.get("caption", payload.get("company", payload.get("source_file", "Unknown")))
                print(f"    extracted: {img_info}")
                top_image_docs.append(img_info)
        except Exception as e:
            print(f"Error processing image result {idx}: {e}")
            continue

    # Combine both lists for routing context
    combined_context = " ,".join(top_text_docs + top_image_docs)
    print(f"Combined vectorstore context for router: {combined_context}")

    llm_router = ChatGroq(model="llama-3.3-70b-versatile")
    question_router = get_question_router_chain(combined_context, llm_router)
    source = question_router.invoke({"question": question})
    print(source)
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, SEARCHING OVER THE INTERNET---"
        )
        return "financial_web_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    messages = state["messages"]
    question = messages[-1].content
    
    documents = state["documents"]
    Intermediate_message = state["Intermediate_message"]

    #  Track retry count in state
    retry_count = state.get("retry_count", 0)
    max_retries = 2  # or 3, depending on how strict you want to be

    llm = ChatOpenAI(model="gpt-4o")
    hallucination_grader = get_hallucination_chain(llm)
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": Intermediate_message}
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
        if retry_count >= max_retries:
            print(f"---MAX RETRIES REACHED ({retry_count}), stopping loop---")
            return "useful"  # fallback → treat as final result instead of looping forever
        else:
            print(f"---DECISION: GENERATION IS NOT GROUNDED, RETRY {retry_count + 1}/{max_retries}---")
            # Increment retry count in state
            state["retry_count"] = retry_count + 1
            return "not supported"

