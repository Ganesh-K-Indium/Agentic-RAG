from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from Graph.invoke_graph import BuildingGraph as RAGGraph

class ManagerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.rag_graph = RAGGraph().get_graph()

    def handle(self, query: str, extra_inputs: dict = None):
        """
        Handle RAG requests directly.
        """
        return self.rag_graph.invoke({"messages": [HumanMessage(content=query)]})
