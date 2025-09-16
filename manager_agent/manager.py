from pydantic import BaseModel
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from Graph.invoke_graph import BuildingGraph as RAGGraph
from IngestionGraph.invoke_graph import IngestionGraph


# ---------- Pydantic schema for ingestion commands ----------
# ManagerAgent.py
class IngestionCommand(BaseModel):
    source: str  # "local_pdf", "confluence", "jira", "sharepoint", "gdrive_folder"
    file_name: Optional[str] = None
    space_key: Optional[str] = None
    ticket_id: Optional[str] = None
    file_url: Optional[str] = None
    folder_id: Optional[str] = None



# ---------- Manager Agent ----------
class ManagerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.rag_graph = RAGGraph().get_graph()
        self.ingestion_graph = IngestionGraph().get_graph()

    def decide_and_route(self, query: str) -> str:
        """
        Decide whether to route to ingestion or RAG.
        """
        decision_prompt = """
        You are a manager agent. Your job is to classify user requests as one of two intents: ingestion or rag.

        - Ingestion: the user wants to add, upload, process, or ingest a document or file into a system 
          (PDF, SharePoint, Jira, Confluence, image store, vector DB).
        - RAG: the user wants to ask a question, summarize, or retrieve knowledge from documents already ingested.

        Return exactly one word: ingestion or rag. 
        Do not include any extra words or explanation.

        Example 1:
        User: "Upload the Q2 report PDF to vector store"
        Intent: ingestion

        Example 2:
        User: "Summarize Pfizer's PDF in vector store and images in image store"
        Intent: rag
        """

        decision = self.llm.invoke(
            [HumanMessage(content=decision_prompt + f"\n\nQuery: {query}")]
        )

        decision_text = decision.content.strip().lower()
        if "ingestion" in decision_text:
            return "ingestion"
        else:
            return "rag"

    def parse_ingestion_command(self, query: str) -> dict:
        """
        Parse ingestion query into structured command with parameters.
        """
        system_prompt =  """
        You are an assistant that extracts structured ingestion commands from user queries.

        Supported sources:
        - local_pdf (requires file_name)
        - confluence (requires space_key)
        - jira (requires ticket_id)
        - sharepoint (requires file_url)
        - gdrive_folder (requires folder_id)

        If some required info is missing, leave that field empty.
        Output must follow the IngestionCommand schema exactly.
        """


        structured_llm = self.llm.with_structured_output(IngestionCommand)

        response = structured_llm.invoke(
            [HumanMessage(content=system_prompt + f"\n\nQuery: {query}")]
        )

        return response.dict()

    def handle(self, query: str, extra_inputs: dict = None):
        """
        Route the request to the correct graph.
        """
        choice = self.decide_and_route(query)

        if choice == "rag":
            return self.rag_graph.invoke({"messages": [HumanMessage(content=query)]})

        elif choice == "ingestion":
            parsed_command = self.parse_ingestion_command(query)

            inputs = {"request": query, "logs": [], **parsed_command}
            if extra_inputs:
                inputs.update(extra_inputs)

            return self.ingestion_graph.invoke(inputs)
