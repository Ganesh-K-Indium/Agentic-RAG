"""
Memory management system for the RAG system to enable faster responses and better context awareness.
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage
import numpy as np
from functools import wraps

class MemoryManager:
    """
    Central memory management for the RAG system.
    Handles caching, conversation memory, and performance optimization.
    """
    
    def __init__(self, cache_ttl: int = 3600, max_cache_size: int = 1000):
        """
        Initialize memory manager.
        
        Args:
            cache_ttl: Time to live for cache entries in seconds (default: 1 hour)
            max_cache_size: Maximum number of cache entries to maintain
        """
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        
        # Initialize memory stores
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.document_cache: Dict[str, Dict[str, Any]] = {}
        self.routing_patterns: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'response_times': [],
            'vectorstore_scores': [],
            'user_satisfaction': []
        }
        
        # Track cache statistics for accurate hit rate calculation
        self.cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.user_preferences: Dict[str, Any] = {
            'preferred_detail_level': 'medium',
            'favorite_companies': [],
            'common_query_types': [],
            'response_format_preference': 'structured'
        }
        
    def generate_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a unique cache key for queries."""
        base_string = query.lower().strip()
        if context:
            base_string += str(sorted(context.items()))
        return hashlib.md5(base_string.encode()).hexdigest()
    
    def is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self.cache_ttl
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        
        # Clean query cache
        expired_keys = [
            key for key, value in self.query_cache.items()
            if not self.is_cache_valid(value.get('timestamp', 0))
        ]
        for key in expired_keys:
            del self.query_cache[key]
            
        # Clean document cache
        expired_keys = [
            key for key, value in self.document_cache.items()
            if not self.is_cache_valid(value.get('timestamp', 0))
        ]
        for key in expired_keys:
            del self.document_cache[key]
    
    def cache_query_result(self, query: str, result: Dict[str, Any], 
                          context: Optional[Dict] = None, quality_score: float = 0.0):
        """Cache query results for faster future responses."""
        cache_key = self.generate_cache_key(query, context)
        
        self.query_cache[cache_key] = {
            'query': query,
            'result': result,
            'context': context,
            'timestamp': time.time(),
            'quality_score': quality_score,
            'access_count': 1
        }
        
        # Maintain cache size limit
        if len(self.query_cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(
                self.query_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            for key, _ in sorted_items[:len(sorted_items) - self.max_cache_size]:
                del self.query_cache[key]
    
    def get_cached_query_result(self, query: str, context: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached query result if available and valid."""
        cache_key = self.generate_cache_key(query, context)
        self.cache_stats['total_requests'] += 1
        
        if cache_key in self.query_cache:
            cached_entry = self.query_cache[cache_key]
            if self.is_cache_valid(cached_entry['timestamp']):
                # Update access count and timestamp
                cached_entry['access_count'] += 1
                cached_entry['last_accessed'] = time.time()
                self.cache_stats['cache_hits'] += 1
                print(f"CACHE HIT for query: {query[:50]}...")
                return cached_entry['result']
            else:
                # Remove expired entry
                del self.query_cache[cache_key]
        
        self.cache_stats['cache_misses'] += 1
        print(f"CACHE MISS for query: {query[:50]}...")
        return None
    
    def cache_document_retrieval(self, query_embedding: List[float], documents: List[Any], 
                               scores: List[float], collection_type: str = "text"):
        """Cache document retrieval results."""
        # Create a hash of the embedding for caching
        embedding_hash = hashlib.md5(str(query_embedding).encode()).hexdigest()
        
        self.document_cache[embedding_hash] = {
            'documents': documents,
            'scores': scores,
            'collection_type': collection_type,
            'timestamp': time.time()
        }
    
    def get_cached_documents(self, query_embedding: List[float], 
                           similarity_threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        """Retrieve cached documents if query embedding is similar enough."""
        embedding_hash = hashlib.md5(str(query_embedding).encode()).hexdigest()
        
        if embedding_hash in self.document_cache:
            cached_entry = self.document_cache[embedding_hash]
            if self.is_cache_valid(cached_entry['timestamp']):
                return cached_entry
        
        # TODO: Implement similarity-based retrieval for similar embeddings
        return None
    
    def learn_routing_pattern(self, query: str, routing_decision: str, 
                            outcome_quality: float, response_time: float):
        """Learn from routing decisions to improve future routing."""
        query_type = self._classify_query_type(query)
        
        if query_type not in self.routing_patterns:
            self.routing_patterns[query_type] = {
                'decisions': [],
                'average_quality': 0.0,
                'average_response_time': 0.0,
                'preferred_route': None
            }
        
        pattern = self.routing_patterns[query_type]
        pattern['decisions'].append({
            'route': routing_decision,
            'quality': outcome_quality,
            'response_time': response_time,
            'timestamp': time.time()
        })
        
        # Update averages and preferred route
        recent_decisions = [d for d in pattern['decisions'] 
                          if time.time() - d['timestamp'] < 86400]  # Last 24 hours
        
        if recent_decisions:
            pattern['average_quality'] = np.mean([d['quality'] for d in recent_decisions])
            pattern['average_response_time'] = np.mean([d['response_time'] for d in recent_decisions])
            
            # Determine preferred route based on quality and speed
            route_performance = {}
            for decision in recent_decisions:
                route = decision['route']
                if route not in route_performance:
                    route_performance[route] = {'quality': [], 'speed': []}
                route_performance[route]['quality'].append(decision['quality'])
                route_performance[route]['speed'].append(decision['response_time'])
            
            # Choose route with best quality-speed balance
            best_route = None
            best_score = 0
            for route, perf in route_performance.items():
                avg_quality = np.mean(perf['quality'])
                avg_speed = 1 / (np.mean(perf['speed']) + 0.1)  # Inverse for speed score
                combined_score = avg_quality * 0.7 + avg_speed * 0.3
                if combined_score > best_score:
                    best_score = combined_score
                    best_route = route
            
            pattern['preferred_route'] = best_route
    
    def get_routing_recommendation(self, query: str) -> Optional[str]:
        """Get routing recommendation based on learned patterns."""
        query_type = self._classify_query_type(query)
        
        if query_type in self.routing_patterns:
            pattern = self.routing_patterns[query_type]
            if pattern['preferred_route'] and pattern['average_quality'] > 0.6:
                return pattern['preferred_route']
        
        return None
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query into types for pattern learning."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'against']):
            return 'comparison'
        elif any(word in query_lower for word in ['revenue', 'profit', 'earnings', 'financial']):
            return 'financial_data'
        elif any(word in query_lower for word in ['risk', 'challenge', 'threat', 'concern']):
            return 'risk_analysis'
        elif any(word in query_lower for word in ['trend', 'growth', 'change', 'over time']):
            return 'trend_analysis'
        elif any(word in query_lower for word in ['current', 'latest', 'recent', 'today']):
            return 'temporal'
        else:
            return 'general'
    
    def update_conversation_memory(self, query: str, response: str, 
                                 context_used: List[str], user_feedback: Optional[float] = None):
        """Update conversation memory with new interaction."""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'context_used': context_used,
            'timestamp': time.time(),
            'user_feedback': user_feedback
        })
        
        # Keep only recent conversation history (last 10 interactions)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_context(self, max_interactions: int = 3) -> List[Dict[str, Any]]:
        """Get recent conversation context for better responses."""
        return self.conversation_history[-max_interactions:] if self.conversation_history else []
    
    def update_user_preferences(self, preference_updates: Dict[str, Any]):
        """Update user preferences based on interactions."""
        self.user_preferences.update(preference_updates)
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights for system optimization."""
        return {
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'average_response_time': np.mean(self.performance_metrics['response_times']) if self.performance_metrics['response_times'] else 0,
            'cache_size': len(self.query_cache),
            'routing_patterns_learned': len(self.routing_patterns),
            'conversation_length': len(self.conversation_history)
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate based on actual requests."""
        total_requests = self.cache_stats['total_requests']
        cache_hits = self.cache_stats['cache_hits']
        return cache_hits / total_requests if total_requests > 0 else 0.0
    
    def initialize_state_memory(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize memory components in graph state."""
        if not state.get('conversation_memory'):
            state['conversation_memory'] = {
                'history': self.get_conversation_context(),
                'user_preferences': self.user_preferences.copy()
            }
        
        if not state.get('query_cache'):
            state['query_cache'] = {}
            
        if not state.get('routing_memory'):
            state['routing_memory'] = {
                'patterns': self.routing_patterns.copy(),
                'recommendation': None
            }
            
        if not state.get('performance_metrics'):
            state['performance_metrics'] = {
                'start_time': time.time(),
                'cache_hits': 0,
                'total_queries': 0
            }
            
        return state


# Decorator for automatic caching and performance tracking
def with_memory(memory_manager: MemoryManager, cache_key_func=None):
    """Decorator to add memory capabilities to graph nodes."""
    def decorator(func):
        @wraps(func)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            
            # Initialize memory components if not present
            state = memory_manager.initialize_state_memory(state)
            
            # Try to get cached result
            if cache_key_func:
                cache_key = cache_key_func(state)
                cached_result = memory_manager.get_cached_query_result(cache_key)
                if cached_result:
                    state['performance_metrics']['cache_hits'] += 1
                    return {**state, **cached_result}
            
            # Execute function
            result = func(state)
            
            # Cache result if applicable
            if cache_key_func and 'documents' in result:
                cache_key = cache_key_func(state)
                memory_manager.cache_query_result(
                    cache_key, 
                    result, 
                    quality_score=len(result.get('documents', []))
                )
            
            # Track performance
            response_time = time.time() - start_time
            memory_manager.performance_metrics['response_times'].append(response_time)
            
            return result
        return wrapper
    return decorator


# Global memory manager instance
global_memory_manager = MemoryManager()