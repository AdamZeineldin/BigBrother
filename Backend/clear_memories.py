#!/usr/bin/env python3
"""
Script to clear all memory nodes from the database
"""
import sys
from db.database import get_memory_nodes, delete_all_memory_nodes

def main(force=False):
    # Get count before deletion
    memories = get_memory_nodes()
    count = len(memories)
    
    if count == 0:
        print("No memory nodes found in database.")
        return
    
    print(f"Found {count} memory node(s) in database.")
    print("\nMemory nodes:")
    for memory in memories:
        print(f"  - ID {memory['id']}: {memory['transcript'][:50]}...")
    
    # Confirm deletion (unless force flag is set)
    if not force:
        response = input(f"\nAre you sure you want to delete all {count} memory node(s)? (yes/no): ").strip().lower()
        if response != "yes":
            print("\n❌ Deletion cancelled.")
            return
    
    deleted_count = delete_all_memory_nodes()
    print(f"\n✅ Successfully deleted {deleted_count} memory node(s).")

if __name__ == "__main__":
    force = "--force" in sys.argv or "-f" in sys.argv
    main(force=force)

