#!/usr/bin/env python3
"""
Test script for the memory recall system
"""
import requests
import json
import sys
from datetime import datetime
from db.database import add_memory_node, get_memory_nodes
from ai.gemini_client2 import recall_memories

BASE_URL = "http://localhost:5000/api"

def create_test_memories():
    """Create some test memory nodes"""
    print("Creating test memory nodes...")
    
    test_memories = [
        {
            "transcript": "I have a meeting tomorrow at 3pm with the team to discuss the project deadline.",
            "audio_path": "test_meeting.wav"
        },
        {
            "transcript": "My name is Adam and I'm working on a new AI memory app called BigBrother.",
            "audio_path": "test_intro.wav"
        },
        {
            "transcript": "The deadline for the project is next Friday. We need to finish the backend API first.",
            "audio_path": "test_deadline.wav"
        },
        {
            "transcript": "I need to buy groceries: milk, eggs, and bread from the store.",
            "audio_path": "test_groceries.wav"
        },
        {
            "transcript": "The weather is really nice today. Perfect for a walk in the park.",
            "audio_path": "test_weather.wav"
        }
    ]
    
    created_ids = []
    for memory in test_memories:
        node_id = add_memory_node(
            transcript=memory["transcript"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            audio_path=memory["audio_path"]
        )
        created_ids.append(node_id)
        print(f"  Created memory node {node_id}: {memory['transcript'][:50]}...")
    
    return created_ids

def test_recall_direct(question, memories):
    """Test recall_memories function directly"""
    print(f"\nAsking: '{question}'")
    print("-" * 60)
    
    try:
        relevant = recall_memories(question, memories, max_results=5)
        
        print(f"Total memories searched: {len(memories)}")
        print(f"Relevant memories found: {len(relevant)}")
        print("\nRelevant memories:")
        
        for i, memory in enumerate(relevant, 1):
            print(f"\n  {i}. Memory ID: {memory['id']}")
            print(f"     Timestamp: {memory['timestamp']}")
            print(f"     Transcript: {memory['transcript']}")
            print(f"     Audio path: {memory.get('audio_path', 'N/A')}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_ask_endpoint(question):
    """Test the ask endpoint via HTTP"""
    print(f"\nAsking via API: '{question}'")
    print("-" * 60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"question": question, "max_results": 5},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Question: {data['question']}")
            print(f"Total memories searched: {data['total_memories_searched']}")
            print(f"Relevant memories found: {data['relevant_memories_found']}")
            print("\nRelevant memories:")
            
            for i, memory in enumerate(data['memories'], 1):
                print(f"\n  {i}. Memory ID: {memory['id']}")
                print(f"     Timestamp: {memory['timestamp']}")
                print(f"     Transcript: {memory['transcript']}")
                print(f"     Audio path: {memory.get('audio_path', 'N/A')}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Make sure Flask is running on port 5000")
    except Exception as e:
        print(f"Error testing ask endpoint: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=" * 60)
    print("Testing Memory Recall System")
    print("=" * 60)
    
    # Check existing memories
    existing = get_memory_nodes()
    print(f"\nExisting memories in database: {len(existing)}")
    
    # Create test memories if needed
    if len(existing) < 3:
        print("\nCreating test memories...")
        create_test_memories()
    else:
        print(f"Using existing {len(existing)} memories")
    
    # Get all memories
    all_memories = get_memory_nodes()
    
    # Test various questions (direct function call first)
    test_questions = [
        "What did I say about the meeting?",
        "When is the project deadline?",
        "What's my name?",
        "What do I need to buy?",
        "How is the weather?"
    ]
    
    print("\n" + "=" * 60)
    print("Testing Direct Function Call (recall_memories)")
    print("=" * 60)
    
    for question in test_questions:
        test_recall_direct(question, all_memories)
        print("\n" + "=" * 60 + "\n")
    
    # Test via API if server is running
    print("\n" + "=" * 60)
    print("Testing via HTTP API")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/memories", timeout=2)
        if response.status_code == 200:
            print("Server is running! Testing API endpoint...\n")
            for question in test_questions[:2]:  # Test first 2 questions via API
                test_ask_endpoint(question)
                print("\n" + "=" * 60 + "\n")
        else:
            print("Server responded with error")
    except:
        print("Server not accessible. Skipping API tests.")
        print("To test API, start server with: python app.py")

if __name__ == "__main__":
    main()

