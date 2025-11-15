import sqlite3
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

def add_memory_node(transcript, timestamp, audio_path=None, video_path=None, video_summary=None):
    """
    Add a new memory node (transcript with timestamp) to the database.
    
    Args:
        transcript: The transcribed text from audio
        timestamp: Timestamp string (e.g., "2024-01-01 12:00:00")
        audio_path: Optional path to associated audio file
        video_path: Optional path to associated video file
        video_summary: Optional text summary of the video
    
    Returns:
        The ID of the newly created memory node
    """
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO memory_nodes (transcript, timestamp, audio_path, video_path, video_summary)
        VALUES (?, ?, ?, ?, ?)
    """, (transcript, timestamp, audio_path, video_path, video_summary))
    conn.commit()
    return cursor.lastrowid

def get_memory_nodes(limit=None):
    """
    Get all memory nodes, ordered by most recent first.
    
    Args:
        limit: Optional limit on number of results
    
    Returns:
        List of dictionaries containing memory node data
    """
    conn = get_connection()
    if limit and limit > 0:
        rows = conn.execute(
            "SELECT * FROM memory_nodes ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM memory_nodes ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(row) for row in rows]

def search_memory_nodes(keyword):
    """
    Search memory nodes by keyword in transcript text and video summary.
    
    Args:
        keyword: Search term to look for in transcripts and video summaries
    
    Returns:
        List of dictionaries containing matching memory nodes
    """
    conn = get_connection()
    search_pattern = f"%{keyword}%"
    rows = conn.execute("""
        SELECT * FROM memory_nodes
        WHERE transcript LIKE ? OR video_summary LIKE ?
        ORDER BY timestamp DESC
    """, (search_pattern, search_pattern)).fetchall()
    return [dict(row) for row in rows]

def get_memory_node_by_id(node_id):
    """
    Get a specific memory node by ID.
    
    Args:
        node_id: The ID of the memory node
    
    Returns:
        Dictionary containing memory node data, or None if not found
    """
    conn = get_connection()
    row = conn.execute("""
        SELECT * FROM memory_nodes WHERE id = ?
    """, (node_id,)).fetchone()
    return dict(row) if row else None

def delete_all_memory_nodes():
    """
    Delete all memory nodes from the database.
    
    Returns:
        Number of memory nodes deleted
    """
    conn = get_connection()
    cursor = conn.execute("DELETE FROM memory_nodes")
    conn.commit()
    return cursor.rowcount

def delete_memory_node(node_id):
    """
    Delete a specific memory node by ID.
    
    Args:
        node_id: The ID of the memory node to delete
    
    Returns:
        True if deleted, False if not found
    """
    conn = get_connection()
    cursor = conn.execute("DELETE FROM memory_nodes WHERE id = ?", (node_id,))
    conn.commit()
    return cursor.rowcount > 0

