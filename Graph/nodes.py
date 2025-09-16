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

    llm = ChatOpenAI(model="gpt-4o-mini")
    company_extractor = get_company_name(llm)
    company = company_extractor.invoke({"question": question})
    print(company)

    filtered_results = [
        doc for doc in results
        if doc.metadata.get("company", "").lower() in company.company.lower()
    ]

    if filtered_results:
        documents.extend(filtered_results)

    tool_call_entry = {
        "tool": "image_retriever"
    }

    print(f"UPDATED DOCUMENTS LENGTH: {len(documents)}")
    return {
        "documents": documents,
        "tool_calls": state.get("tool_calls", []) + [tool_call_entry]
    }


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

    filtered_docs = []
    llm_grade_document = ChatOpenAI(model="gpt-4o")
    retrieval_grader = get_retrival_grader_chain(llm_grade_document)

    results_log = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        results_log.append({"doc": d.page_content, "grade": grade})
        if grade.lower() == "yes":
            filtered_docs.append(d)

    tool_call_entry = {
        "tool": "retrieval_grader"
    }

    print(f"FILTERED DOCS COUNT: {len(filtered_docs)}")
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
