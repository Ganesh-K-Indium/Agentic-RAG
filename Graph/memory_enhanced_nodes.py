"""
Memory-enhanced nodes for the RAG system that leverage caching and conversation context.
"""

from Graph.memory_manager import global_memory_manager, with_memory
import time
from typing import Dict, Any

def memory_enhanced_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced retrieve function with memory capabilities.
    """
    print("---MEMORY-ENHANCED RETRIEVE---")
    start_time = time.time()
    
    messages = state["messages"]
    question = messages[-1].content
    
    # Get session-specific memory manager
    session_memory_manager = global_memory_manager  # Default fallback
    if 'session_metadata' in state and 'user_id' in state['session_metadata']:
        # Use session-specific memory from session manager
        from Graph.session_aware_wrapper import global_session_manager_v2
        session_id = state['session_metadata'].get('session_id', 'default')
        session_graph = global_session_manager_v2.active_sessions.get(session_id)
        if session_graph and hasattr(session_graph, 'session_memory_manager'):
            session_memory_manager = session_graph.session_memory_manager
            print(f"Using session memory manager for: {session_id}")
    
    # Quick cache check with timeout
    try:
        # Check conversation context for related queries (limited to recent 2 for speed)
        conversation_context = session_memory_manager.get_conversation_context()
        context_queries = [ctx['query'] for ctx in conversation_context[-2:]]  # Reduced from 3 to 2
        
        # Generate cache key based on question and recent context
        cache_context = {'recent_queries': context_queries} if context_queries else None
        cached_result = session_memory_manager.get_cached_query_result(question, cache_context)
        print(f"Cache lookup for '{question}': {'HIT' if cached_result else 'MISS'}")
        print(f"Cache size: {len(session_memory_manager.query_cache)}")
    except Exception as e:
        print(f"Cache lookup failed, proceeding without cache: {e}")
        cached_result = None
    
    if cached_result:
        print("---USING CACHED RETRIEVAL RESULTS---")
        state['performance_metrics']['cache_hits'] += 1
        return {**state, **cached_result}
    
    # If not cached, perform normal retrieval
    from Graph.nodes import retrieve
    result = retrieve(state)
    
    # Cache the results (with error handling for performance)
    try:
        if result.get('documents'):
            quality_score = len(result['documents']) / 4.0  # Normalize to 0-1 based on max expected docs
            session_memory_manager.cache_query_result(
                question, 
                result, 
                cache_context, 
                quality_score
            )
            print(f"Cached query result with key: {session_memory_manager.generate_cache_key(question, cache_context)}")
    except Exception as e:
        print(f"Caching failed, continuing without cache: {e}")
    
    # Track performance
    response_time = time.time() - start_time
    try:
        session_memory_manager.performance_metrics['response_times'].append(response_time)
    except Exception as e:
        print(f"Performance tracking failed: {e}")
    
    return result


def memory_enhanced_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced generate function that uses conversation memory for better responses.
    """
    print("---MEMORY-ENHANCED GENERATE---")
    
    # Get session-specific memory manager
    session_memory_manager = global_memory_manager  # Default fallback
    if 'session_metadata' in state:
        from Graph.session_aware_wrapper import global_session_manager_v2
        session_id = state['session_metadata'].get('session_id', 'default')
        session_graph = global_session_manager_v2.active_sessions.get(session_id)
        if session_graph and hasattr(session_graph, 'session_memory_manager'):
            session_memory_manager = session_graph.session_memory_manager
    
    # Get conversation context for continuity
    conversation_context = session_memory_manager.get_conversation_context()
    if conversation_context:
        # Add conversation context to the generation process
        context_summary = []
        for ctx in conversation_context[-2:]:  # Last 2 interactions
            context_summary.append(f"Previous Q: {ctx['query'][:100]}...")
            if ctx.get('user_feedback') and ctx['user_feedback'] > 0.7:
                context_summary.append("(User found this helpful)")
        
        if context_summary:
            print(f"Using conversation context: {context_summary}")
            # Add context to state for generation
            state['conversation_context'] = context_summary
    
    # Check user preferences for response formatting
    user_prefs = session_memory_manager.user_preferences
    if user_prefs.get('preferred_detail_level'):
        state['detail_level'] = user_prefs['preferred_detail_level']
    if user_prefs.get('response_format_preference'):
        state['format_preference'] = user_prefs['response_format_preference']
    
    # Perform generation (import here to avoid circular imports)
    from Graph.nodes import generate
    result = generate(state)
    
    # Update conversation memory
    if result.get('Intermediate_message'):
        context_used = [doc.metadata.get('source_file', 'unknown') for doc in state.get('documents', [])]
        session_memory_manager.update_conversation_memory(
            state['messages'][-1].content,
            result['Intermediate_message'],
            context_used
        )
    
    return result


def memory_enhanced_grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced document grading that learns from user feedback and past decisions.
    """
    print("---MEMORY-ENHANCED DOCUMENT GRADING---")
    
    # Get session-specific memory manager
    session_memory_manager = global_memory_manager  # Default fallback
    if 'session_metadata' in state:
        from Graph.session_aware_wrapper import global_session_manager_v2
        session_id = state['session_metadata'].get('session_id', 'default')
        session_graph = global_session_manager_v2.active_sessions.get(session_id)
        if session_graph and hasattr(session_graph, 'session_memory_manager'):
            session_memory_manager = session_graph.session_memory_manager
            
    # Get conversation context to understand document relevance patterns
    conversation_context = session_memory_manager.get_conversation_context()
    recent_feedback = [ctx.get('user_feedback', 0) for ctx in conversation_context if ctx.get('user_feedback')]
    
    # If recent feedback is positive, be more inclusive with similar document types
    if recent_feedback and sum(recent_feedback) / len(recent_feedback) > 0.7:
        print("---APPLYING LENIENT GRADING BASED ON POSITIVE FEEDBACK---")
        # This would be implemented in the actual grading logic
        state['grading_mode'] = 'lenient'
    
    # Perform normal document grading
    from Graph.nodes import grade_documents
    result = grade_documents(state)
    
    # Learn from grading patterns
    if result.get('documents'):
        grading_quality = len(result['documents']) / len(state.get('documents', [1]))
        # This could be used to improve future grading thresholds
        try:
            session_memory_manager.performance_metrics['vectorstore_scores'].append(grading_quality)
        except Exception as e:
            print(f"Performance tracking failed: {e}")
    
    return result


def finalize_with_memory_update(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final step that updates memory with the complete interaction (optimized for speed).
    """
    print("---FINALIZING WITH MEMORY UPDATE---")
    
    # Quick memory update without expensive operations
    try:
        # Learn routing patterns (simplified and fast)
        if state.get('routing_memory'):
            routing_info = state['routing_memory']
            quality_estimate = 0.8 if state.get('documents') else 0.3
            routing_decision = routing_info.get('decision', 'unknown')
            
            # Get session-specific memory manager
            session_memory_manager = global_memory_manager  # Default fallback
            if 'session_metadata' in state:
                from Graph.session_aware_wrapper import global_session_manager_v2
                session_id = state['session_metadata'].get('session_id', 'default')
                session_graph = global_session_manager_v2.active_sessions.get(session_id)
                if session_graph and hasattr(session_graph, 'session_memory_manager'):
                    session_memory_manager = session_graph.session_memory_manager
                    
            if routing_decision != 'unknown' and hasattr(session_memory_manager, 'learn_routing_pattern'):
                session_memory_manager.learn_routing_pattern(
                    state['messages'][-1].content,
                    routing_decision,
                    quality_estimate
                )
    except Exception as e:
        print(f"Memory update failed, continuing: {e}")
    
    return state