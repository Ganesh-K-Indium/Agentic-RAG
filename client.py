import requests
import json
from dotenv import load_dotenv
import os
import datetime

# Load environment variables
load_dotenv()

# API endpoint for the new memory-enhanced server
url = "http://localhost:8001/ask"

def test_memory_enhanced_server():
    """Test the new memory-enhanced RAG server with session management."""
    
    # Example query with user ID for session management
    payload = {
        "query": "tell me about meta's revenue by user geography based on estimate of the geography",
        "user_id": "john_doe_analyst",  # Unique user identifier for session tracking
        "extra_inputs": {}  # Optional additional parameters
    }
    
    print("🚀 Testing Memory-Enhanced RAG Server")
    print("=" * 50)
    print(f"Query: {payload['query']}")
    print(f"User ID: {payload['user_id']}")
    print(f"Endpoint: {url}")
    print("\n⏳ Processing query...")
    
    try:
        # Send request to the server
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ Response received successfully!")
            print("=" * 50)
            
            # Display the answer
            print("\n📄 ANSWER:")
            print("-" * 30)
            answer = data.get('answer', 'No answer provided')
            if isinstance(answer, dict):
                print(json.dumps(answer, indent=2))
            else:
                print(answer)
            
            # Display session information
            print("\n👤 SESSION INFO:")
            print("-" * 30)
            session_info = data.get('session_info', {})
            print(f"Session ID: {session_info.get('session_id', 'N/A')}")
            print(f"User ID: {session_info.get('user_id', 'N/A')}")
            print(f"Conversation Length: {session_info.get('conversation_length', 0)}")
            print(f"Cache Hit Rate: {session_info.get('cache_hit_rate', 0):.2%}")
            
            # Display processing information
            print("\n🔧 PROCESSING INFO:")
            print("-" * 30)
            print(f"Documents Used: {data.get('documents_used', 0)}")
            print(f"Routing Decision: {data.get('routing_decision', 'Unknown')}")
            
            return data
            
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("\n⏰ Request timed out. The server might be processing a complex query.")
        return None
    except requests.exceptions.ConnectionError:
        print("\n🚨 Connection error. Make sure the server is running on port 8001.")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return None

def test_session_endpoint(user_id):
    """Test the session management endpoint."""
    session_url = f"http://localhost:8001/session/{user_id}"
    
    print(f"\n🔍 Testing Session Endpoint for user: {user_id}")
    print("-" * 40)
    
    try:
        response = requests.get(session_url, timeout=10)
        
        if response.status_code == 200:
            session_data = response.json()
            print("✅ Session data retrieved:")
            print(json.dumps(session_data, indent=2))
            return session_data
        else:
            print(f"❌ Error retrieving session: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the main query endpoint
    result = test_memory_enhanced_server()
    
    if result:
        # Test the session endpoint
        test_session_endpoint("john_doe_analyst")
        
        print("\n🎉 Client test completed successfully!")
        print("\n💡 Try running this script multiple times to see memory caching in action!")
    else:
        print("\n⚠️ Client test failed. Check if the server is running.")
        print("Start the server with: python -m uvicorn app:app --reload --port 8001")
    