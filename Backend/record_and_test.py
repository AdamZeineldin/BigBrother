#!/usr/bin/env python3
"""
Interactive script to record audio and test the recall system
"""
import requests
import time
import sys

# Use port 5001 to avoid AirPlay conflict
PORT = 5001
BASE_URL = f"http://localhost:{PORT}/api"

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"http://localhost:{PORT}/", timeout=2)
        return True
    except:
        return False

def record_audio():
    """Record audio interactively"""
    print("\n" + "="*60)
    print("RECORDING AUDIO")
    print("="*60)
    print("\nStarting recording...")
    print("(The recording will continue until you press Enter)")
    
    # Start recording (will auto-generate unique filename)
    try:
        response = requests.post(
            f"{BASE_URL}/record/start",
            json={},  # Let server generate unique filename
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"Error starting recording: {response.json()}")
            return None
            
        print("✅ Recording started! Speak now...")
        print("Press Enter when done recording...")
        input()
        
        # Stop recording (this will auto-transcribe and save as memory node)
        print("\nStopping recording and transcribing...")
        response = requests.post(
            f"{BASE_URL}/record/stop",
            json={"auto_transcribe": True},
            timeout=60  # Transcription can take time
        )
        
        if response.status_code == 200:
            data = response.json()
            audio_path = data.get('output_path', data.get('audio_path', 'N/A'))
            memory_id = data.get('id', 'N/A')
            
            print("\n✅ Recording saved as memory node!")
            print(f"   Memory ID: {memory_id}")
            print(f"   Transcript: {data.get('transcript', 'N/A')[:100]}...")
            print(f"   Audio path: {audio_path}")
            if memory_id != 'N/A':
                print(f"   Play audio: {BASE_URL.replace('/api', '')}/api/memories/{memory_id}/audio")
            return data
        else:
            print(f"Error: {response.json()}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to server!")
        print("Make sure Flask is running: python app.py")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

def ask_question():
    """Ask a question and get relevant memories"""
    print("\n" + "="*60)
    print("ASK A QUESTION")
    print("="*60)
    
    question = input("\nEnter your question: ").strip()
    
    if not question:
        print("No question entered.")
        return
    
    print(f"\nSearching for: '{question}'...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"question": question, "max_results": 5},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Found {data['relevant_memories_found']} relevant memories")
            print(f"   (Searched {data['total_memories_searched']} total memories)\n")
            
            if data['memories']:
                for i, memory in enumerate(data['memories'], 1):
                    memory_id = memory['id']
                    audio_path = memory.get('audio_path', 'N/A')
                    print(f"{i}. Memory ID {memory_id} ({memory['timestamp']})")
                    print(f"   Transcript: {memory['transcript'][:150]}...")
                    print(f"   Audio path: {audio_path}")
                    print(f"   Play: {BASE_URL.replace('/api', '')}/api/memories/{memory_id}/audio\n")
            else:
                print("No relevant memories found.")
        else:
            print(f"Error: {response.json()}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to server!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def list_memories():
    """List all memories"""
    print("\n" + "="*60)
    print("ALL MEMORIES")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/memories?limit=10", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            memories = data.get('memories', [])
            
            if memories:
                print(f"\nFound {len(memories)} memories:\n")
                for memory in memories:
                    print(f"ID {memory['id']}: {memory['timestamp']}")
                    print(f"  {memory['transcript'][:100]}...\n")
            else:
                print("\nNo memories found.")
        else:
            print(f"Error: {response.json()}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to server!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    print("="*60)
    print("MEMORY RECALL SYSTEM - Interactive Test")
    print("="*60)
    
    # Check server
    if not check_server():
        print("\n❌ Server is not running!")
        print(f"\nTo start the server, run:")
        print("  cd backend")
        print("  python3 app.py")
        print(f"\nServer will run on port {PORT}")
        sys.exit(1)
    
    print("\n✅ Server is running!")
    
    while True:
        print("\n" + "="*60)
        print("OPTIONS:")
        print("  1. Record audio (creates memory node)")
        print("  2. Ask a question (recall memories)")
        print("  3. List all memories")
        print("  4. Exit")
        print("="*60)
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "1":
            record_audio()
        elif choice == "2":
            ask_question()
        elif choice == "3":
            list_memories()
        elif choice == "4":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please enter 1-4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)

