"This module contains all info about about the nodes in the graph"
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_tavily import TavilySearch
from load_vector_dbs.prompts_and_chains import (get_retrival_grader_chain, get_rag_chain,
                                                          get_company_name, get_question_rewriter_chain)
from load_vector_dbs.load_dbs import load_vector_database
load_dotenv()

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    messages = state["messages"]
    question = messages[-1].content
    init = load_vector_database()
    retriever, _, _ = init.get_text_retriever()
    documents = retriever.invoke(question)
    print(f"DOCUMENTS RETRIEVED AND NUMBER OF DOCUMENTS ARE {len(documents)}")
    return {"documents": documents}

def retrieve_from_images_data(state):
    """
    Retrieve documents from image_retriever

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): append new documents to documents state key retrieved by image retriever.
    """
    messages = state["messages"]
    question = messages[-1].content
    documents = state["documents"]
    init = load_vector_database()
    _, image_retriever, _ = init.get_image_retriever()
    results = image_retriever.invoke(question)
    llm = ChatOpenAI(model="gpt-4o-mini")
    company_extractor = get_company_name(llm)
    company = company_extractor.invoke({"question":question})
    print(company)
    filtered_results = [doc for doc in results if doc.metadata.get("company").lower() in company.company.lower()]
    print(f"DOCUMENTS RETRIEVED FROM IMAGE AND NUMBER OF DOCUMENTS ARE {len(filtered_results)}")
    if len(filtered_results)>0:
        for doc in filtered_results:
            documents.append(doc)
    
    print(f"UPDATING NEW DOCUMENTS IN STATE NOW LENGTH OF RELAVENT DOCUMENTS ARE {len(documents)}")
    return {"documents": documents}


def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[-1].content
    documents = state["documents"]

    llm = ChatOpenAI(model="gpt-4o-mini")
    rag_chain = get_rag_chain(llm)
    Intermediate_message = rag_chain.invoke(
        {"documents": documents, "question": question}
    )

    retry_count = state.get("retry_count", 0)
    return {"Intermediate_message": Intermediate_message, "retry_count": retry_count + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    messages = state["messages"]
    question = messages[-1].content
    documents = state["documents"]
    print(f"TOTAL NUMBER OF DOCUMENTS ARE {len(documents)}")
    # Score each doc
    filtered_docs = []
    llm_grade_document = ChatOpenAI(model="gpt-4o")
    retrieval_grader = get_retrival_grader_chain(llm_grade_document)
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        print(grade)
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue        
        
    print(f"LENGTH OF DOCUMENTS ARE {len(filtered_docs)}")
    return {"documents": filtered_docs}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[-1].content

    # Re-write question
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    question_rewriter = get_question_rewriter_chain(llm)
    better_question = question_rewriter.invoke({"question": question})
    return {"messages": [better_question]}

def web_search(state):
    print("---WEB SEARCH---")
    messages = state["messages"]
    question = messages[-1].content
    web_search_tool = TavilySearch(k=3)
    docs = web_search_tool.invoke({"query": question}) # or however you're calling the API

    # Normalize docs into strings
    if isinstance(docs, list):
        web_results = "\n".join(
            d["content"] if isinstance(d, dict) and "content" in d else str(d)
            for d in docs
        )
    else:
        web_results = str(docs)

    return {
        **state,
        "documents": [Document(page_content=web_results)],  # keep consistent with retriever
    }


def financial_web_search(state):
    print("---WEB SEARCH---")
    messages = state["messages"]
    question = messages[-1].content

    web_search_tool = TavilySearch(k=3)
    docs = web_search_tool.invoke({"query": question})

    # Normalize docs into text
    if isinstance(docs, list):
        web_results = "\n".join(
            d.get("content", "") if isinstance(d, dict) else str(d)
            for d in docs
        )
    else:
        web_results = str(docs)

    # ✅ Always return a list of Documents
    return {"documents": [Document(page_content=web_results)]}  

def show_result(state):
    """
    Show result will be used to set the final result in the state graph and finally end the process
    
    Args:
        state (dict): The current graph state

    Returns:a
        state (dict): Updates the final answer in the messages
    """
    Final_answer = AIMessage(content = state["Intermediate_message"])
    print(f'SHOWING THE RESULTS: {Final_answer}')
    return {"messages":Final_answer}   