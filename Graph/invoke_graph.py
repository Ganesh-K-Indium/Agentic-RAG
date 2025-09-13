"This module is useful for building the graph which will create an agentic workflow."
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from Graph.graph_state import GraphState
from Graph.nodes import (retrieve_from_images_data, web_search, retrieve,
                         grade_documents, generate, transform_query,
                         financial_web_search, show_result)
from Graph.edges import (route_question, decide_to_generate,
                         grade_generation_v_documents_and_question)
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

class BuildingGraph:
    """
    This class has one class method which is responsible for building the graph
    """
    def get_graph(self):
        """
        This class method is responsible for creating the graph
        Returns:-
        app :- complied graph
        """
        workflow = StateGraph(GraphState)

        workflow.add_node("image_analyses_retrival",retrieve_from_images_data)
        workflow.add_node("web_search", web_search)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("transform_query", transform_query)
        workflow.add_node("financial_web_search", financial_web_search)
        workflow.add_node("show_result", show_result)

        workflow.add_conditional_edges(
            START,
            route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve"
            },
        )

        workflow.add_edge("retrieve", "image_analyses_retrival")

        workflow.add_edge("image_analyses_retrival", "grade_documents")

        workflow.add_edge("web_search", "generate")

        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "financial_web_search": "financial_web_search",
                "generate": "generate",
            },
        )

        workflow.add_edge("financial_web_search", "generate")
        workflow.add_edge("transform_query", "retrieve")

        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": "show_result",
                "not useful": "transform_query",
            },
        )
        workflow.add_edge("show_result", END)

        app = workflow.compile()
        return app

# graph_obj = BuildingGraph()
# agent = graph_obj.get_graph()
if __name__ == '__main__':
    
    inputs = {
    "messages": ["""tell me about the distribution of discovery projects across various 
                 phases of the R&D pipeline along with the timeline and number of projects for pfizer?"""]
    }
    graph_obj = BuildingGraph()
    agent = graph_obj.get_graph()
    messages = agent.invoke(inputs)
    print(messages["messages"][1].content)
    