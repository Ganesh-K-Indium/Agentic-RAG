"This module is useful for building the graph which will create an agentic workflow."
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from Graph.graph_state import GraphState
from Graph.nodes import (retrieve_from_images_data, web_search, retrieve,
                         grade_documents, generate, transform_query,
                         financial_web_search, show_result, integrate_web_search,
                         evaluate_vectorstore_quality, analyze_cross_reference_needs,
                         determine_summary_strategy, categorize_documents_by_source,
                         generate_with_cross_reference_and_citations)
from Graph.edges import (route_question, decide_to_generate,
                         grade_generation_v_documents_and_question,
                         decide_after_web_integration, decide_cross_reference_approach,
                         decide_after_cross_reference_analysis)
from Graph.session_aware_wrapper import SessionAwareGraphWrapper
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

class BuildingGraph:
    """
    This class has one class method which is responsible for building the graph
    """
    def __init__(self, session_id: str = None):
        """
        Initialize graph builder with session awareness.
        
        Args:
            session_id: Unique identifier for the user session
        """
        self.session_id = session_id
        
    def get_graph(self):
        """
        This class method is responsible for creating the graph
        Returns:-
        app :- complied graph
        """
        workflow = StateGraph(GraphState)

        # Import memory-enhanced nodes
        from Graph.memory_enhanced_nodes import (
            memory_enhanced_retrieve, 
            memory_enhanced_generate, 
            memory_enhanced_grade_documents,
            finalize_with_memory_update
        )
        
        workflow.add_node("image_analyses_retrival",retrieve_from_images_data)
        workflow.add_node("web_search", web_search)
        workflow.add_node("retrieve", memory_enhanced_retrieve)  # Memory-enhanced
        workflow.add_node("grade_documents", memory_enhanced_grade_documents)  # Memory-enhanced
        workflow.add_node("generate", memory_enhanced_generate)  # Memory-enhanced
        workflow.add_node("transform_query", transform_query)
        workflow.add_node("financial_web_search", financial_web_search)
        workflow.add_node("show_result", show_result)
        workflow.add_node("integrate_web_search", integrate_web_search)
        workflow.add_node("evaluate_vectorstore_quality", evaluate_vectorstore_quality)
        workflow.add_node("analyze_cross_reference", analyze_cross_reference_needs)
        workflow.add_node("determine_strategy", determine_summary_strategy)
        workflow.add_node("categorize_documents", categorize_documents_by_source)
        workflow.add_node("generate_with_citations", generate_with_cross_reference_and_citations)
        workflow.add_node("finalize_memory", finalize_with_memory_update)  # New memory finalization node

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
                "generate": "analyze_cross_reference",  # First analyze if cross-referencing is needed
                "integrate_web_search": "integrate_web_search",
            },
        )

        # New edge for web integration
        workflow.add_conditional_edges(
            "integrate_web_search",
            decide_after_web_integration,
            {
                "grade_documents": "grade_documents",
                "financial_web_search": "financial_web_search",
            },
        )

        # Cross-referencing workflow edges
        workflow.add_conditional_edges(
            "analyze_cross_reference",
            decide_after_cross_reference_analysis,
            {
                "categorize_documents": "categorize_documents",
                "generate": "generate",
            },
        )

        workflow.add_edge("categorize_documents", "determine_strategy")
        workflow.add_edge("determine_strategy", "generate_with_citations")

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

        # Add similar grading for enhanced generation with citations
        workflow.add_conditional_edges(
            "generate_with_citations",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate_with_citations",
                "useful": "show_result",
                "not useful": "transform_query",
            },
        )
        workflow.add_edge("show_result", "finalize_memory")
        workflow.add_edge("finalize_memory", END)

        app = workflow.compile()
        
        # Wrap the compiled app to inject session context
        if self.session_id:
            return SessionAwareGraphWrapper(app, self.session_id)
        else:
            return app

# graph_obj = BuildingGraph()
# agent = graph_obj.get_graph()
if __name__ == '__main__':
    # Example with session-aware execution
    from Graph.session_aware_wrapper import global_session_manager_v2
    import time
    
    # Create a session-specific graph
    user_session_id = f"demo_user_{int(time.time())}"
    print(f"Creating session: {user_session_id}")
    
    # Get session-aware graph
    session_graph = global_session_manager_v2.get_or_create_session_graph(
        user_session_id, BuildingGraph
    )
    
    inputs = {
        "messages": ["""tell me about the distribution of discovery projects across various 
                     phases of the R&D pipeline along with the timeline and number of projects for pfizer?"""],
        "vectorstore_searched": False,
        "web_searched": False,
        "vectorstore_quality": "none",
        "needs_web_fallback": False,
        "retry_count": 0,
        "tool_calls": [],
        "cross_reference_analysis": {},
        "document_sources": {},
        "citation_info": [],
        "summary_strategy": "single_source"
    }
    
    # Execute with session context
    messages = session_graph.invoke(inputs)
    print(messages["messages"][1].content)
    
    # Show session information
    session_summary = session_graph.get_session_summary()
    print(f"\nSession Summary:")
    print(f"Session ID: {session_summary['session_id']}")
    print(f"Memory Manager ID: {session_summary['memory_manager_id']}")
    print(f"Conversation Length: {session_summary['conversation_length']}")
    