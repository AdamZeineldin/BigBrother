#!/usr/bin/env python3
"""
Test script for MemoryNode system and Gemini search functionality.

This script:
1. Creates sample MemoryNodes for testing
2. Tests retrieving MemoryNodes
3. Tests Gemini search to find the most relevant MemoryNode based on queries
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the Backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.database import (
    init_db,
    create_memory_node,
    get_memory_nodes,
    get_all_memory_nodes_for_search,
    get_memory_node_by_id
)
from ai.gemini_client import search_memory_nodes


def create_sample_memory_nodes():
    """Create sample MemoryNodes for testing"""
    print("Creating sample MemoryNodes...")
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    recordings_dir = data_dir / "recordings"
    audio_dir = data_dir / "audio"
    transcripts_dir = data_dir / "transcripts"
    
    # Create directories if they don't exist
    recordings_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    
    sample_nodes = [
        {
            "file_path": str(recordings_dir / "motion_20250116_120000.mp4"),
            "file_type": "recording",
            "timestamp": "2025-01-16T12:00:00",
            "metadata": {
                "video_path": str(recordings_dir / "motion_20250116_120000.mp4"),
                "audio_path": str(audio_dir / "motion_20250116_120000.wav"),
                "transcript_path": str(transcripts_dir / "motion_20250116_120000.txt"),
                "summary": "A person enters the room and sits at a desk. They start working on a computer.",
                "transcript": "Hello, I'm starting my work session. I need to finish the project by today.",
                "objects_detected": ["person", "chair", "laptop"],
                "description": "Person working at desk",
                "thumbnail_path": str(data_dir / "images" / "thumbnail_20250116_120000.jpg")
            }
        },
        {
            "file_path": str(recordings_dir / "motion_20250116_130000.mp4"),
            "file_type": "recording",
            "timestamp": "2025-01-16T13:00:00",
            "metadata": {
                "video_path": str(recordings_dir / "motion_20250116_130000.mp4"),
                "audio_path": str(audio_dir / "motion_20250116_130000.wav"),
                "transcript_path": str(transcripts_dir / "motion_20250116_130000.txt"),
                "summary": "A cat walks across the room and jumps onto a window sill.",
                "transcript": "The weather is nice today. I should take a break and go outside.",
                "objects_detected": ["cat", "window"],
                "description": "Cat on window sill",
                "thumbnail_path": str(data_dir / "images" / "thumbnail_20250116_130000.jpg")
            }
        },
        {
            "file_path": str(recordings_dir / "motion_20250116_140000.mp4"),
            "file_type": "recording",
            "timestamp": "2025-01-16T14:00:00",
            "metadata": {
                "video_path": str(recordings_dir / "motion_20250116_140000.mp4"),
                "audio_path": str(audio_dir / "motion_20250116_140000.wav"),
                "transcript_path": str(transcripts_dir / "motion_20250116_140000.txt"),
                "summary": "Someone is cooking in the kitchen. They are preparing a meal with vegetables.",
                "transcript": "I'm making dinner. Let me chop these vegetables.",
                "objects_detected": ["person", "kitchen", "vegetables"],
                "description": "Cooking in kitchen",
                "thumbnail_path": str(data_dir / "images" / "thumbnail_20250116_140000.jpg")
            }
        },
    ]
    
    created_ids = []
    for node_data in sample_nodes:
        try:
            node_id = create_memory_node(
                file_path=node_data["file_path"],
                file_type=node_data["file_type"],
                timestamp=node_data["timestamp"],
                metadata=json.dumps(node_data["metadata"])
            )
            created_ids.append(node_id)
            print(f"  ✓ Created MemoryNode {node_id}: {node_data['file_type']} - {Path(node_data['file_path']).name}")
        except Exception as e:
            print(f"  ✗ Failed to create MemoryNode: {e}")
    
    print(f"\nCreated {len(created_ids)} MemoryNodes\n")
    return created_ids


def test_get_memory_nodes():
    """Test retrieving MemoryNodes"""
    print("=" * 60)
    print("Testing MemoryNode Retrieval")
    print("=" * 60)
    
    # Get all nodes
    all_nodes = get_memory_nodes()
    print(f"\n1. Total MemoryNodes: {len(all_nodes)}")
    
    # Get by file type
    for file_type in ["recording", "video", "audio", "transcript"]:
        nodes = get_memory_nodes(file_type=file_type)
        print(f"   {file_type.capitalize()} nodes: {len(nodes)}")
    
    # Get with limit
    limited = get_memory_nodes(limit=3)
    print(f"\n2. Limited to 3 nodes: {len(limited)}")
    
    # Show sample node
    if all_nodes:
        sample = all_nodes[0]
        print(f"\n3. Sample MemoryNode (ID: {sample['id']}):")
        print(f"   File Type: {sample['file_type']}")
        print(f"   File Path: {sample['file_path']}")
        print(f"   Timestamp: {sample['timestamp']}")
        if sample.get('metadata'):
            try:
                metadata = json.loads(sample['metadata'])
                print(f"   Metadata: {json.dumps(metadata, indent=6)}")
            except:
                print(f"   Metadata: {sample['metadata']}")
    print()


def test_gemini_search():
    """Test Gemini search functionality"""
    print("=" * 60)
    print("Testing Gemini Search for Most Relevant MemoryNode")
    print("=" * 60)
    
    # Get all MemoryNodes for search
    all_nodes = get_all_memory_nodes_for_search()
    
    if not all_nodes:
        print("\n⚠ No MemoryNodes found. Creating sample nodes...\n")
        create_sample_memory_nodes()
        all_nodes = get_all_memory_nodes_for_search()
    
    if not all_nodes:
        print("❌ No MemoryNodes available for search")
        return
    
    print(f"\nSearching through {len(all_nodes)} MemoryNodes\n")
    
    # Test queries
    test_queries = [
        "Find videos with people in them",
        "Show me transcripts about work or projects",
        "Get the video where someone is cooking",
        "Find anything related to a cat",
        "What happened around 1 PM?",
        "Show me the most recent video",
        "Find transcripts that mention weather",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: '{query}'")
        print("-" * 60)
        
        try:
            results = search_memory_nodes(
                query=query,
                memory_nodes=all_nodes,
                max_results=3
            )
            
            if results:
                print(f"   Found {len(results)} relevant MemoryNode(s):\n")
                for j, node in enumerate(results, 1):
                    print(f"   [{j}] MemoryNode ID: {node['id']}")
                    print(f"       Type: {node['file_type']}")
                    print(f"       Path: {Path(node['file_path']).name}")
                    print(f"       Timestamp: {node['timestamp']}")
                    
                    # Show relevant metadata
                    if node.get('metadata'):
                        try:
                            metadata = json.loads(node['metadata']) if isinstance(node['metadata'], str) else node['metadata']
                            if 'summary' in metadata and metadata.get('summary'):
                                print(f"       Summary: {metadata['summary'][:80]}...")
                            if 'transcript' in metadata and metadata.get('transcript'):
                                print(f"       Transcript: {metadata['transcript'][:80]}...")
                            if 'video_path' in metadata and metadata.get('video_path'):
                                print(f"       Video: {Path(metadata['video_path']).name}")
                            if 'audio_path' in metadata and metadata.get('audio_path'):
                                print(f"       Audio: {Path(metadata['audio_path']).name}")
                        except:
                            pass
                    print()
            else:
                print("   No relevant MemoryNodes found\n")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}\n")
        
        print()


def test_get_most_relevant():
    """Test getting the single most relevant MemoryNode"""
    print("=" * 60)
    print("Testing: Get Most Relevant MemoryNode")
    print("=" * 60)
    
    all_nodes = get_all_memory_nodes_for_search()
    
    if not all_nodes:
        print("\n⚠ No MemoryNodes found. Creating sample nodes...\n")
        create_sample_memory_nodes()
        all_nodes = get_all_memory_nodes_for_search()
    
    queries = [
        "Find the video with a person working",
        "Get the transcript about the weather",
        "Show me the cooking video",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        try:
            results = search_memory_nodes(
                query=query,
                memory_nodes=all_nodes,
                max_results=1  # Get only the most relevant one
            )
            
            if results:
                node = results[0]
                print(f"✓ Most Relevant MemoryNode:")
                print(f"  ID: {node['id']}")
                print(f"  Type: {node['file_type']}")
                print(f"  File: {Path(node['file_path']).name}")
                print(f"  Timestamp: {node['timestamp']}")
                
                # Get full node details
                full_node = get_memory_node_by_id(node['id'])
                if full_node and full_node.get('metadata'):
                    try:
                        metadata = json.loads(full_node['metadata'])
                        print(f"  Full Metadata:")
                        print(f"    {json.dumps(metadata, indent=4)}")
                    except:
                        pass
            else:
                print("  No relevant MemoryNode found")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print()


def interactive_search():
    """Interactive search mode - let user input their own queries"""
    print("=" * 60)
    print("Interactive MemoryNode Search")
    print("=" * 60)
    print("\nEnter your search queries. Type 'quit' or 'exit' to stop.\n")
    
    all_nodes = get_all_memory_nodes_for_search()
    
    if not all_nodes:
        print("⚠ No MemoryNodes found. Creating sample nodes...\n")
        create_sample_memory_nodes()
        all_nodes = get_all_memory_nodes_for_search()
    
    if not all_nodes:
        print("❌ No MemoryNodes available for search")
        return
    
    print(f"Searching through {len(all_nodes)} MemoryNodes\n")
    print("-" * 60)
    
    while True:
        try:
            # Get user query
            query = input("\n🔍 What are you looking for? (or 'quit' to exit): ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive search...\n")
                break
            
            print(f"\nSearching for: '{query}'")
            print("-" * 60)
            
            # Ask for number of results
            try:
                max_results_input = input("How many results? (default: 3): ").strip()
                max_results = int(max_results_input) if max_results_input else 3
            except ValueError:
                max_results = 3
            
            # Perform search
            try:
                results = search_memory_nodes(
                    query=query,
                    memory_nodes=all_nodes,
                    max_results=max_results
                )
                
                if results:
                    print(f"\n✓ Found {len(results)} relevant MemoryNode(s):\n")
                    for i, node in enumerate(results, 1):
                        print(f"[{i}] MemoryNode ID: {node['id']}")
                        print(f"    Type: {node['file_type']}")
                        print(f"    Path: {Path(node['file_path']).name}")
                        print(f"    Timestamp: {node['timestamp']}")
                        
                        # Show relevant metadata
                        if node.get('metadata'):
                            try:
                                metadata = json.loads(node['metadata']) if isinstance(node['metadata'], str) else node['metadata']
                                
                                if 'summary' in metadata and metadata.get('summary'):
                                    print(f"    Summary: {metadata['summary'][:100]}...")
                                
                                if 'transcript' in metadata and metadata.get('transcript'):
                                    transcript_preview = metadata['transcript'][:100]
                                    print(f"    Transcript: {transcript_preview}...")
                                
                                if 'video_path' in metadata and metadata.get('video_path'):
                                    print(f"    Video: {Path(metadata['video_path']).name}")
                                
                                if 'audio_path' in metadata and metadata.get('audio_path'):
                                    print(f"    Audio: {Path(metadata['audio_path']).name}")
                                
                                if 'objects_detected' in metadata and metadata.get('objects_detected'):
                                    objects = metadata['objects_detected']
                                    if isinstance(objects, list) and objects:
                                        print(f"    Objects: {', '.join(objects)}")
                            except Exception as e:
                                pass
                        print()
                else:
                    print("\n❌ No relevant MemoryNodes found for that query.")
                    
            except Exception as e:
                print(f"\n❌ Search failed: {e}")
                import traceback
                traceback.print_exc()
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive search...\n")
            break
        except EOFError:
            print("\n\nExiting interactive search...\n")
            break


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("MemoryNode System Test Suite")
    print("=" * 60 + "\n")
    
    # Initialize database
    print("Initializing database...")
    try:
        init_db()
        print("✓ Database initialized\n")
    except Exception as e:
        print(f"⚠ Database initialization warning: {e}\n")
    
    # Check if we should create sample nodes
    existing_nodes = get_memory_nodes()
    if len(existing_nodes) == 0:
        print("No existing MemoryNodes found.")
        response = input("Create sample MemoryNodes for testing? (y/n): ").strip().lower()
        if response == 'y':
            create_sample_memory_nodes()
    else:
        print(f"Found {len(existing_nodes)} existing MemoryNodes\n")
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Run automated tests")
    print("2. Interactive search (enter your own queries)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3, default: 2): ").strip() or "2"
    
    try:
        if choice == "1":
            # Run automated tests
            test_get_memory_nodes()
            test_gemini_search()
            test_get_most_relevant()
            
            print("=" * 60)
            print("✓ All tests completed!")
            print("=" * 60 + "\n")
            
        elif choice == "2":
            # Interactive search only
            interactive_search()
            
        elif choice == "3":
            # Both
            test_get_memory_nodes()
            print("\n")
            interactive_search()
            
        else:
            print("Invalid choice. Running interactive search...\n")
            interactive_search()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

