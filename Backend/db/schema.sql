-- Memory nodes table: stores audio transcripts with timestamps and optional video/audio paths
-- Each row is a "memory node" that can be searched/recalled by AI
CREATE TABLE IF NOT EXISTS memory_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    audio_path TEXT,
    video_path TEXT,
    video_summary TEXT
);

-- Index for faster text search
CREATE INDEX IF NOT EXISTS idx_transcript ON memory_nodes(transcript);

