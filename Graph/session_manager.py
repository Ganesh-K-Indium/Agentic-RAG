"""
Session management for persistent memory across multiple conversations.
"""

import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from Graph.memory_manager import MemoryManager

class SessionManager:
    """
    Manages persistent memory sessions for users.
    """
    
    def __init__(self, session_dir: str = "sessions", auto_save_interval: int = 300):
        """
        Initialize session manager.
        
        Args:
            session_dir: Directory to store session files
            auto_save_interval: Auto-save interval in seconds
        """
        self.session_dir = session_dir
        self.auto_save_interval = auto_save_interval
        self.current_session_id = None
        self.memory_manager = None
        
        # Create session directory if it doesn't exist
        os.makedirs(session_dir, exist_ok=True)
    
    def create_session(self, user_id: str = "default") -> str:
        """Create a new session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{user_id}_{timestamp}"
        
        self.current_session_id = session_id
        self.memory_manager = MemoryManager()
        
        # Save initial session
        self.save_session()
        
        return session_id
    
    def load_session(self, session_id: str) -> bool:
        """Load an existing session."""
        session_file = os.path.join(self.session_dir, f"{session_id}.pkl")
        
        if os.path.exists(session_file):
            try:
                with open(session_file, 'rb') as f:
                    session_data = pickle.load(f)
                
                self.current_session_id = session_id
                self.memory_manager = session_data['memory_manager']
                
                print(f"Loaded session {session_id}")
                return True
            except Exception as e:
                print(f"Error loading session {session_id}: {e}")
                return False
        
        return False
    
    def save_session(self):
        """Save current session to disk."""
        if not self.current_session_id or not self.memory_manager:
            return False
        
        session_file = os.path.join(self.session_dir, f"{self.current_session_id}.pkl")
        session_data = {
            'session_id': self.current_session_id,
            'memory_manager': self.memory_manager,
            'last_saved': datetime.now(),
            'session_metadata': {
                'queries_count': len(self.memory_manager.conversation_history),
                'cache_size': len(self.memory_manager.query_cache),
                'learned_patterns': len(self.memory_manager.routing_patterns)
            }
        }
        
        try:
            with open(session_file, 'wb') as f:
                pickle.dump(session_data, f)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def list_sessions(self, user_id: str = None) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions = []
        
        for filename in os.listdir(self.session_dir):
            if filename.endswith('.pkl'):
                session_id = filename[:-4]  # Remove .pkl extension
                
                if user_id and not session_id.startswith(user_id):
                    continue
                
                session_file = os.path.join(self.session_dir, filename)
                try:
                    with open(session_file, 'rb') as f:
                        session_data = pickle.load(f)
                    
                    sessions.append({
                        'session_id': session_id,
                        'last_saved': session_data.get('last_saved'),
                        'metadata': session_data.get('session_metadata', {})
                    })
                except Exception as e:
                    print(f"Error reading session {session_id}: {e}")
        
        # Sort by last saved time
        sessions.sort(key=lambda x: x.get('last_saved', datetime.min), reverse=True)
        return sessions
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Remove sessions older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0
        
        for filename in os.listdir(self.session_dir):
            if filename.endswith('.pkl'):
                session_file = os.path.join(self.session_dir, filename)
                try:
                    with open(session_file, 'rb') as f:
                        session_data = pickle.load(f)
                    
                    last_saved = session_data.get('last_saved')
                    if last_saved and last_saved < cutoff_date:
                        os.remove(session_file)
                        cleaned_count += 1
                except Exception as e:
                    print(f"Error processing session {filename}: {e}")
        
        print(f"Cleaned up {cleaned_count} old sessions")
        return cleaned_count
    
    def get_memory_manager(self) -> Optional[MemoryManager]:
        """Get the current memory manager."""
        return self.memory_manager
    
    def initialize_for_graph(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize session memory in graph state."""
        if not self.memory_manager:
            self.memory_manager = MemoryManager()
        
        # Initialize memory in state
        state = self.memory_manager.initialize_state_memory(state)
        
        # Add session metadata
        state['session_metadata'] = {
            'session_id': self.current_session_id,
            'session_start_time': datetime.now().isoformat(),
            'auto_save_enabled': True
        }
        
        return state


# Global session manager
global_session_manager = SessionManager()


def with_session_management(func):
    """Decorator to add session management to graph execution."""
    def wrapper(*args, **kwargs):
        # Auto-save session periodically
        if global_session_manager.memory_manager:
            query_count = len(global_session_manager.memory_manager.conversation_history)
            if query_count > 0 and query_count % 5 == 0:  # Save every 5 queries
                global_session_manager.save_session()
        
        result = func(*args, **kwargs)
        return result
    
    return wrapper


def create_user_session(user_id: str = "default") -> str:
    """Create a new user session."""
    return global_session_manager.create_session(user_id)


def load_user_session(session_id: str) -> bool:
    """Load an existing user session."""
    success = global_session_manager.load_session(session_id)
    if success and global_session_manager.memory_manager:
        # Update global memory manager reference
        from Graph.memory_manager import global_memory_manager
        global_memory_manager.__dict__.update(global_session_manager.memory_manager.__dict__)
    return success


def save_current_session() -> bool:
    """Save the current session."""
    return global_session_manager.save_session()


def get_session_summary() -> Dict[str, Any]:
    """Get a summary of the current session."""
    if not global_session_manager.memory_manager:
        return {"error": "No active session"}
    
    return {
        "session_id": global_session_manager.current_session_id,
        "conversation_length": len(global_session_manager.memory_manager.conversation_history),
        "cache_size": len(global_session_manager.memory_manager.query_cache),
        "learned_patterns": len(global_session_manager.memory_manager.routing_patterns),
        "performance_insights": global_session_manager.memory_manager.get_performance_insights(),
        "user_preferences": global_session_manager.memory_manager.user_preferences
    }