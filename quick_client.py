#!/usr/bin/env python3
"""
Simple client example for quick testing of the Memory-Enhanced RAG Server
"""

import requests

def quick_test(query="What is Tesla's main business?", user_id="quick_test_user"):
    """Quick test function for the RAG server."""
    try:
        response = requests.post(
            "http://localhost:8001/ask",
            json={"query": query, "user_id": user_id},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query: {query}")
            print(f"👤 User: {user_id}")
            print(f"📊 Documents: {data.get('documents_used', 0)}")
            print(f"🔀 Route: {data.get('routing_decision', 'unknown')}")
            print(f"💬 Session: {data.get('session_info', {}).get('conversation_length', 0)} queries")
            print(f"\n📄 Answer:\n{data.get('answer', 'No answer')}")
            return data
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    # Allow command line query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        user_id = "cli_user"
    else:
        query = "What is Meta's consolidated balance sheet for the year 2023?"
        user_id = "hello"
    
    quick_test(query, user_id)