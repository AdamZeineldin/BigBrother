import sqlite3
import logging
import os

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "db", "schema.sql")

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # Ensure data folder exists
    data_dir = os.path.join(BASE_DIR, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    conn = get_connection()
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()

def add_event(event_type, timestamp, description, image_path=None, audio_path=None):
    conn = get_connection()
    conn.execute("""
        INSERT INTO events (type, timestamp, description, image_path, audio_path)
        VALUES (?, ?, ?, ?, ?)
    """, (event_type, timestamp, description, image_path, audio_path))
    conn.commit()

def get_events():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM events ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]

def search_events(keyword):
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM events
        WHERE description LIKE ?
        ORDER BY id DESC
    """, (f"%{keyword}%",)).fetchall()
    return [dict(row) for row in rows]

# MemoryNode functions
def create_memory_node(file_path, file_type, timestamp, metadata=None):
    """
    Create a new MemoryNode in the database.
    
    Args:
        file_path: Path to the file (video, audio, or transcript)
        file_type: Type of file ('video', 'audio', or 'transcript')
        timestamp: Timestamp when the file was created
        metadata: Optional JSON string containing additional metadata
    
    Returns:
        The ID of the created MemoryNode
    """
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO memory_nodes (file_path, file_type, timestamp, metadata)
        VALUES (?, ?, ?, ?)
    """, (file_path, file_type, timestamp, metadata))
    conn.commit()
    return cursor.lastrowid

def get_memory_nodes(file_type=None, limit=None):
    """
    Get all MemoryNodes, optionally filtered by file_type.
    
    Args:
        file_type: Optional filter by file type ('video', 'audio', or 'transcript')
        limit: Optional limit on number of results
    
    Returns:
        List of MemoryNode dictionaries
    """
    conn = get_connection()
    if file_type:
        query = "SELECT * FROM memory_nodes WHERE file_type = ? ORDER BY timestamp DESC"
        params = (file_type,)
    else:
        query = "SELECT * FROM memory_nodes ORDER BY timestamp DESC"
        params = ()
    
    if limit:
        query += f" LIMIT {limit}"
    
    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]

def get_memory_node_by_id(node_id):
    """
    Get a MemoryNode by its ID.
    
    Args:
        node_id: The ID of the MemoryNode
    
    Returns:
        MemoryNode dictionary or None if not found
    """
    conn = get_connection()
    row = conn.execute("SELECT * FROM memory_nodes WHERE id = ?", (node_id,)).fetchone()
    return dict(row) if row else None

def get_memory_nodes_by_timestamp_range(start_timestamp, end_timestamp):
    """
    Get MemoryNodes within a timestamp range.
    
    Args:
        start_timestamp: Start timestamp (ISO format string)
        end_timestamp: End timestamp (ISO format string)
    
    Returns:
        List of MemoryNode dictionaries
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM memory_nodes
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp DESC
    """, (start_timestamp, end_timestamp)).fetchall()
    return [dict(row) for row in rows]

def get_all_memory_nodes_for_search():
    """
    Get all MemoryNodes formatted for Gemini search.
    Returns a list of dictionaries with id, file_path, file_type, timestamp, and metadata.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, file_path, file_type, timestamp, metadata
        FROM memory_nodes
        ORDER BY timestamp DESC
    """).fetchall()
    return [dict(row) for row in rows]

def update_memory_node_metadata(node_id, metadata):
    """
    Update the metadata of a MemoryNode by ID.
    
    Args:
        node_id: The ID of the MemoryNode to update
        metadata: New metadata (will be JSON stringified)
    
    Returns:
        True if successful, False otherwise
    """
    import json
    conn = get_connection()
    try:
        metadata_str = json.dumps(metadata) if not isinstance(metadata, str) else metadata
        conn.execute("""
            UPDATE memory_nodes
            SET metadata = ?
            WHERE id = ?
        """, (metadata_str, node_id))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Failed to update MemoryNode metadata: {e}")
        return False

def get_memory_node_by_file_path(file_path):
    """
    Get a MemoryNode by its file_path.
    
    Args:
        file_path: The file path of the MemoryNode
    
    Returns:
        MemoryNode dictionary or None if not found
    """
    conn = get_connection()
    row = conn.execute("SELECT * FROM memory_nodes WHERE file_path = ?", (file_path,)).fetchone()
    return dict(row) if row else None


def cleanup_orphaned_memory_nodes():
    """
    Remove memory nodes whose associated files (video, audio, transcript) no longer exist.
    Checks all files referenced in the metadata as well.
    
    Returns:
        Tuple of (deleted_count, list of deleted node IDs)
    """
    import json
    conn = get_connection()
    deleted_ids = []
    
    try:
        # Get all memory nodes
        rows = conn.execute("SELECT * FROM memory_nodes").fetchall()
        
        for row in rows:
            node = dict(row)
            node_id = node['id']
            file_path = node.get('file_path', '')
            
            # Parse metadata to check all referenced files
            metadata = {}
            try:
                metadata_str = node.get('metadata', '{}') or '{}'
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            except:
                pass
            
            # Check if primary file_path exists
            primary_exists = os.path.exists(file_path) if file_path else False
            
            # Check all files referenced in metadata
            video_path = metadata.get('video_path')
            audio_path = metadata.get('audio_path')
            transcript_path = metadata.get('transcript_path')
            thumbnail_path = metadata.get('thumbnail_path')
            
            # If we have video_path in metadata, prefer that over file_path
            primary_path = video_path or file_path
            
            # Check if any required files exist
            video_exists = os.path.exists(video_path) if video_path else False
            audio_exists = os.path.exists(audio_path) if audio_path else False
            transcript_exists = os.path.exists(transcript_path) if transcript_path else False
            
            # Delete the node if:
            # 1. Primary path doesn't exist, OR
            # 2. Video path is specified but doesn't exist, OR
            # 3. Both video and transcript don't exist (both are typically required for a complete event)
            should_delete = False
            
            if primary_path:
                # If we have a primary path, check if it exists
                if not os.path.exists(primary_path):
                    should_delete = True
            else:
                # If no primary path, check if we have any valid files
                # If we have a video_path in metadata but it doesn't exist, delete
                if video_path and not video_exists:
                    should_delete = True
                # If no video_path but no files exist, delete
                elif not video_path and not (audio_exists or transcript_exists):
                    should_delete = True
            
            if should_delete:
                conn.execute("DELETE FROM memory_nodes WHERE id = ?", (node_id,))
                deleted_ids.append(node_id)
                logging.info(f"Deleted orphaned MemoryNode {node_id} (files missing: primary={primary_path}, video={video_path}, audio={audio_path}, transcript={transcript_path})")
        
        conn.commit()
        return len(deleted_ids), deleted_ids
        
    except Exception as e:
        logging.error(f"Error cleaning up orphaned memory nodes: {e}", exc_info=True)
        conn.rollback()
        return 0, []

