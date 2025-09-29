"""
Session-aware graph wrapper that properly manages memory across different user sessions.
"""

from typing import Dict, Any, Optional
import time
from Graph.session_manager import global_session_manager, load_user_session
from Graph.memory_manager import MemoryManager

class SessionAwareGraphWrapper:
    """
    Wrapper that ensures each graph execution is tied to a specific user session.
    This is how the system knows which session the interaction belongs to.
    """
    
    def __init__(self, compiled_graph, session_id: str):
        """
        Initialize session-aware wrapper.
        
        Args:
            compiled_graph: The compiled LangGraph workflow
            session_id: Unique identifier for this user session
        """
        self.compiled_graph = compiled_graph
        self.session_id = session_id
        self.session_memory_manager = None
        
        # Load or create session-specific memory
        self._initialize_session()
    
    def _initialize_session(self):
        """
        Initialize or load the session-specific memory manager.
        This is where session differentiation happens.
        """
        # Try to load existing session
        if load_user_session(self.session_id):
            print(f"Loaded existing session: {self.session_id}")
            self.session_memory_manager = global_session_manager.get_memory_manager()
        else:
            # Create new session
            print(f"Creating new session: {self.session_id}")
            global_session_manager.create_session(self.session_id)
            self.session_memory_manager = global_session_manager.get_memory_manager()
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the graph with session-specific context.
        This is where the session context gets injected into every execution.
        """
        # Inject session context into the state
        session_enhanced_inputs = self._prepare_session_context(inputs)
        
        # Execute the graph with optimized recursion limit for speed
        start_time = time.time()
        config = {"recursion_limit": 35}  # Reduced from 50 to 35 for faster completion
        result = self.compiled_graph.invoke(session_enhanced_inputs, config=config)
        execution_time = time.time() - start_time
        
        # Post-process with session-specific learning
        self._post_process_session_learning(session_enhanced_inputs, result, execution_time)
        
        return result
    
    def _prepare_session_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject session-specific context into the graph state.
        This is how each node knows which session it's operating in.
        """
        # Clone inputs to avoid modifying original
        session_inputs = inputs.copy()
        
        # Initialize session-specific memory components
        if self.session_memory_manager:
            session_inputs = self.session_memory_manager.initialize_state_memory(session_inputs)
        
        # Add session metadata that travels with the state
        session_inputs['session_metadata'] = {
            'session_id': self.session_id,
            'session_start_time': time.time(),
            'memory_manager_id': id(self.session_memory_manager),  # Unique ID for this session's memory
            'user_id': self.session_id.split('_')[0] if '_' in self.session_id else 'default'
        }
        
        # Add session-specific conversation context
        if self.session_memory_manager:
            conversation_context = self.session_memory_manager.get_conversation_context()
            session_inputs['conversation_memory'] = {
                'history': conversation_context,
                'user_preferences': self.session_memory_manager.user_preferences.copy(),
                'session_stats': {
                    'total_queries': len(self.session_memory_manager.conversation_history),
                    'cache_size': len(self.session_memory_manager.query_cache),
                    'learned_patterns': len(self.session_memory_manager.routing_patterns)
                }
            }
        
        print(f"Prepared session context for: {self.session_id}")
        print(f"  - Memory manager ID: {id(self.session_memory_manager)}")
        print(f"  - Conversation history: {len(conversation_context) if conversation_context else 0} items")
        
        return session_inputs
    
    def _post_process_session_learning(self, inputs: Dict[str, Any], 
                                     result: Dict[str, Any], execution_time: float):
        """
        Learn from this interaction and update session-specific memory.
        """
        if not self.session_memory_manager:
            return
        
        # Extract query and response for learning
        messages = inputs.get('messages', [])
        if messages:
            query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            response = result.get('Intermediate_message', '')
            
            # Update conversation memory with session-specific context
            context_used = []
            if result.get('documents'):
                context_used = [
                    doc.metadata.get('source_file', 'unknown') 
                    for doc in result['documents'] 
                    if hasattr(doc, 'metadata')
                ]
            
            self.session_memory_manager.update_conversation_memory(
                query, response, context_used
            )
            
            # Learn routing patterns specific to this session
            if result.get('routing_memory'):
                routing_info = result['routing_memory']
                quality_estimate = 0.8 if result.get('documents') else 0.3
                
                # Get routing decision safely
                routing_decision = routing_info.get('decision', 'unknown')
                if routing_decision == 'unknown':
                    # Infer routing decision from state
                    if result.get('vectorstore_searched') and result.get('web_searched'):
                        routing_decision = 'vectorstore_with_web_supplement'
                    elif result.get('vectorstore_searched'):
                        routing_decision = 'vectorstore'
                    elif result.get('web_searched'):
                        routing_decision = 'web_search'
                
                self.session_memory_manager.learn_routing_pattern(
                    query,
                    routing_decision,
                    quality_estimate,
                    execution_time
                )
            
            print(f"Updated session {self.session_id} with new learning")
        
        # Auto-save session periodically
        query_count = len(self.session_memory_manager.conversation_history)
        if query_count > 0 and query_count % 3 == 0:  # Save every 3 queries
            global_session_manager.save_session()
            print(f"Auto-saved session {self.session_id}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of this specific session."""
        if not self.session_memory_manager:
            return {"error": "No session memory"}
        
        return {
            "session_id": self.session_id,
            "memory_manager_id": id(self.session_memory_manager),
            "conversation_length": len(self.session_memory_manager.conversation_history),
            "cache_size": len(self.session_memory_manager.query_cache),
            "learned_patterns": len(self.session_memory_manager.routing_patterns),
            "user_preferences": self.session_memory_manager.user_preferences,
            "performance_insights": self.session_memory_manager.get_performance_insights()
        }


class SessionManager:
    """
    Manages multiple concurrent user sessions.
    This is how the system differentiates between different users/conversations.
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, SessionAwareGraphWrapper] = {}
    
    def get_or_create_session_graph(self, session_id: str, graph_builder_class):
        """
        Get existing session graph or create new one.
        Each session gets its own graph wrapper with isolated memory.
        """
        if session_id not in self.active_sessions:
            # Create new session-specific graph
            builder = graph_builder_class(session_id=session_id)
            session_graph = builder.get_graph()
            self.active_sessions[session_id] = session_graph
            print(f"Created new session graph for: {session_id}")
        else:
            print(f"Using existing session graph for: {session_id}")
        
        return self.active_sessions[session_id]
    
    def list_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all currently active sessions with their stats."""
        return {
            session_id: graph.get_session_summary() 
            for session_id, graph in self.active_sessions.items()
        }
    
    def cleanup_inactive_sessions(self, max_idle_time: int = 3600):
        """Remove sessions that haven't been used recently."""
        # This would be implemented based on last activity timestamps
        pass


# Global session manager for handling multiple concurrent sessions
global_session_manager_v2 = SessionManager()