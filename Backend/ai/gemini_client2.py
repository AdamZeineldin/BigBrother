"""
Gemini client for AI-powered memory recall
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def get_api_key():
    """Load API key from environment"""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Make sure your .env file contains: GOOGLE_API_KEY=your_key_here"
        )
    return key


def recall_memories(question: str, memory_nodes: list, model: str = "gemini-2.5-flash", max_results: int = 5):
    """
    Use Gemini to find relevant memory nodes based on a question.
    
    Args:
        question: The user's question
        memory_nodes: List of memory node dictionaries from the database
        model: Gemini model to use
        max_results: Maximum number of relevant memories to return
    
    Returns:
        List of relevant memory node dictionaries
    """
    if not memory_nodes:
        return []
    
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    
    # Format memory nodes for Gemini
    memories_text = "MEMORY NODES:\n\n"
    for i, node in enumerate(memory_nodes):
        memory_text = f"Memory {i+1} (ID: {node.get('id', 'N/A')}):\n"
        memory_text += f"Timestamp: {node.get('timestamp', 'N/A')}\n"
        
        if node.get('transcript'):
            memory_text += f"Transcript: {node['transcript']}\n"
        
        if node.get('video_summary'):
            memory_text += f"Video Summary: {node['video_summary']}\n"
        
        memory_text += "\n"
        memories_text += memory_text
    
    # Create prompt for Gemini
    prompt = f"""You are a memory recall assistant. Given a user's question and a list of memory nodes (audio transcripts and video summaries), identify which memory nodes are relevant to answering the question.

User's Question: {question}

{memories_text}

Instructions:
1. Analyze the question and understand what information the user is looking for
2. Review each memory node and determine its relevance to the question
3. Return ONLY a comma-separated list of memory numbers (1, 2, 3, etc.) that are relevant to the question
4. Return at most {max_results} memory numbers, ordered by relevance (most relevant first)
5. If no memories are relevant, return "none"
6. Only return the numbers, nothing else

Example response format: 3, 7, 2
"""
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
        )
        
        result_text = (response.text or "").strip().lower()
        
        # Parse the response
        if result_text == "none" or not result_text:
            return []
        
        # Extract memory numbers
        memory_indices = []
        for num_str in result_text.split(","):
            num_str = num_str.strip()
            try:
                idx = int(num_str) - 1  # Convert to 0-based index
                if 0 <= idx < len(memory_nodes):
                    memory_indices.append(idx)
            except ValueError:
                continue
        
        # Return the relevant memory nodes
        relevant_memories = [memory_nodes[i] for i in memory_indices if i < len(memory_nodes)]
        return relevant_memories
        
    except Exception as e:
        # If Gemini fails, fall back to keyword search
        print(f"Gemini recall failed: {e}", file=__import__('sys').stderr)
        return []

