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

