from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from Graph.invoke_graph import BuildingGraph as RAGGraph
from Graph.session_aware_wrapper import global_session_manager_v2
import time
from typing import Optional, Dict, Any

class ManagerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        # Remove direct graph creation - we'll use session-specific graphs
        
    def handle(self, query: str, user_id: str = "anonymous", extra_inputs: dict = None):
        """
        Handle RAG requests with session awareness.
        
        Args:
            query: The user's question
            user_id: Unique identifier for the user (for session management)
            extra_inputs: Additional inputs for the graph
        
        Returns:
            dict: Response from the RAG system with session information
        """
        # Create or get session-specific graph
        session_graph = global_session_manager_v2.get_or_create_session_graph(
            self._get_or_create_session_id(user_id), RAGGraph
        )
        
        # Prepare inputs with session context
        inputs = {
            "messages": [HumanMessage(content=query)],
            "vectorstore_searched": False,
            "web_searched": False,
            "vectorstore_quality": "none",
            "needs_web_fallback": False,
            "retry_count": 0,
            "tool_calls": [],
            "cross_reference_analysis": {},
            "document_sources": {},
            "citation_info": [],
            "summary_strategy": "standard"
        }
        
        # Add extra inputs if provided
        if extra_inputs:
            inputs.update(extra_inputs)
        
        # Execute with session context
        result = session_graph.invoke(inputs)
        
        # Add session information to response
        session_summary = session_graph.get_session_summary()
        result['session_info'] = {
            'session_id': session_summary['session_id'],
            'conversation_length': session_summary['conversation_length'],
            'cache_hit_rate': session_summary.get('performance_insights', {}).get('cache_hit_rate', 0),
            'user_id': user_id
        }
        
        return result
    
    def _get_or_create_session_id(self, user_id: str) -> str:
        """
        Create a consistent session ID for the user that persists for longer periods.
        This ensures better caching across multiple queries from the same user.
        """
        # Create a session ID that lasts for 24 hours (86400 seconds)
        # This gives much better caching while still allowing session rotation
        timestamp = int(time.time())
        return f"{user_id}_{timestamp // 86400}"  # New session every 24 hours
    
    def get_user_session_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get session summary for a specific user.
        """
        session_id = self._get_or_create_session_id(user_id)
        active_sessions = global_session_manager_v2.list_active_sessions()
        
        return active_sessions.get(session_id, {"error": "No active session found"})
    
    def clear_user_session(self, user_id: str) -> bool:
        """
        Clear/reset a user's session.
        """
        session_id = self._get_or_create_session_id(user_id)
        # This would require implementing session clearing in the session manager
        # For now, we'll just create a new session
        try:
            # Force create new session by using current timestamp
            new_session_id = f"{user_id}_{int(time.time())}"
            global_session_manager_v2.get_or_create_session_graph(new_session_id, RAGGraph)
            return True
        except Exception:
            return False
