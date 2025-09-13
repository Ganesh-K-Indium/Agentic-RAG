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
    vectorstore_source_files = init.get_vector_store_files(vectorstore)
    print(vectorstore_source_files)
    llm_router = ChatGroq(model="llama-3.3-70b-versatile")
    question_router = get_question_router_chain(vectorstore_source_files, llm_router)
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
        answer_score = answer_grader.invoke({"question": question, "generation": Intermediate_message})
        answer_grade = answer_score.binary_score
        if answer_grade.lower() == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"