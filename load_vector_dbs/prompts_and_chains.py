"This module contains all the chains that will be usefull in building the nodes of the graph"

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic_models.models import (RouteQuery, GradeDocuments, 
                                        GradeHallucinations, ExtractCompany,GradeAnswer)


def get_question_router_chain(vectorstore_source_files, llm_router):
    structured_llm_router = llm_router.with_structured_output(RouteQuery)

    SYSTEM_PROMPT = f"""You are an expert at routing a user question to a vectorstore or web search. The vectorstore contains documents related to the 10k-financial report file of {vectorstore_source_files}.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""
    print(SYSTEM_PROMPT)
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    question_router = route_prompt | structured_llm_router
    return question_router

def get_retrival_grader_chain(llm_grade_document):
    structured_llm_grader = llm_grade_document.with_structured_output(GradeDocuments)

    SYSTEM_PROMPT_GRADE = """You are a grader assessing the relevance of a retrieved document to a user question.  

        - If the document **explicitly** contains **keywords** or **semantic meaning** related to the user question, it is **highly relevant**.  
        - If it contains **partial information** but lacks some context, it is **moderately relevant**.  
        - If it does not contain relevant information, it is **not relevant**.  

        ### **Special Case if document started with "This is an image with the caption:":**
        - If the document starts with **"This is an image with the caption:"**, then give **extra priority** to the document.
        - Read the **entire document carefully** to see if **any keywords** related to the user's query appear.
        - Ignore the company name like jp morgon, pfizer, google etc in the question and then decide whether the document is relevant or not. 

        ### **Scoring System:**
        - **Return "Yes" ** if the document is a strong match or even if the document is partially relevant.  
        - **Return "No" ** if it is not relevant at all.  
        """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_GRADE),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def get_rag_chain(llm_generate):
    prompt = """### Task:You are an AI assistant integrated into a Retrieval-Augmented Generation (RAG) system. When a user asks a question, you must generate a relevant answer. Additionally, you should suggest alternative or follow-up questions the user may ask based on the context of their original question and relavant Context.

                ### Instructions:
                1. **Answer the User's Question**: Retrieve the most relevant information from the knowledge base and generate a clear and concise response.
                2. **Generate Suggested Questions**:
                - Provide 3 to 5 related questions that the user might find useful.
                - Ensure the suggestions are contextually relevant and expand on the topic.
                - Avoid redundant or overly generic questions.
                3. **Formatting**:
                - Present the answer first.
                - List the suggested questions under a "Related Questions" section.
                - Format the response in a user-friendly and readable manner.
                """
    RAG_Prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", "Relavant Context: \n\n {documents} \n\n Question: {question} \n\nAnswer:"),
        ]
    )
    rag_chain = RAG_Prompt | llm_generate | StrOutputParser()
    return rag_chain

def get_hallucination_chain(llm_grade_hallucination):
    llm_hallucination_grader = llm_grade_hallucination.with_structured_output(GradeHallucinations)

    SYSTEM_PROMPT_GRADE_HALLUCINATION = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_GRADE_HALLUCINATION),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | llm_hallucination_grader
    
    return hallucination_grader

def get_company_name(llm):
    llm_company_extractor = llm.with_structured_output(ExtractCompany)

    SYSTEM_PROMPT = """you are an expert who can identify the comapny name from the given user question 
    and map it to one of the below company. 
    1. amazon
    2. berkshire
    3. google
    4. Jhonson and Jhonosn
    5. jp morgan
    6. meta
    7. microsoft
    8. nvidia
    9. tesla
    10. visa
    11. walmart
    12. pfizer
    
    ## Instructions
    - make sure you should only generate the comapny name nothing else.
    - if user is asking about companies in a short form or abbriviations like jpmc, you should be able to map it with jp morgon
    - Strictly do not change the company spellings, keep them as it is as mentioned above.
    """

    company_name_extraction_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "User's Question: \n\n {question} \n\n company: "),
        ]
    )

    company_name_extractor = company_name_extraction_prompt | llm_company_extractor
    
    return company_name_extractor

def get_answer_quality_chain(llm_answer_grade):
    llm_answer_grader = llm_answer_grade.with_structured_output(GradeAnswer)

    SYSTEM_ANSWER_GRADER = """
You are a strict but fair evaluator. Decide if the assistant’s answer resolves the user’s question.
Output ONLY "yes" or "no".

Grading Rules:
- "yes" → The answer is correct, mostly correct, or sufficiently addresses the question (even if minor details are missing).
- "no" → The answer is irrelevant, factually incorrect, or does not attempt to answer the question.

Examples:
Q: Show the financial performance of a Company in the last 5 years.
A: Company’s revenue has grown steadily. 2019: $280B, 2020: $386B, 2021: $470B, 2022: $514B, 2023: $575B.
Grade: yes

Q: Show the financial performance of Company in the last 5 years.
A: Company is a tech company that sells books and cloud services.
Grade: no
"""

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_ANSWER_GRADER),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | llm_answer_grader
    return answer_grader

def get_question_rewriter_chain(llm):
    SYSTEM_QUESTION_REWRITER = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_QUESTION_REWRITER),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ] 
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter
