from flask import Blueprint, request, jsonify, send_file
import os
from datetime import datetime

from db.database import (
    add_memory_node,
    get_memory_nodes,
    search_memory_nodes,
    get_memory_node_by_id
)
from audio import recorder, transcribe_audio, save_transcript
from ai.gemini_client2 import recall_memories

api = Blueprint("api", __name__)

@api.route("/memories", methods=["GET"])
def get_memories():
    """Get all memory nodes, optionally limited"""
    limit = request.args.get("limit", type=int)
    memories = get_memory_nodes(limit=limit)
    return jsonify({"memories": memories}), 200

@api.route("/memories/<int:node_id>", methods=["GET"])
def get_memory(node_id):
    """Get a specific memory node by ID"""
    memory = get_memory_node_by_id(node_id)
    if memory:
        return jsonify({"memory": memory}), 200
    return jsonify({"error": "Memory node not found"}), 404

@api.route("/search", methods=["GET"])
def search():
    """Search memory nodes by keyword"""
    q = request.args.get("q", "")
    if not q:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    memories = search_memory_nodes(q)
    return jsonify({"memories": memories}), 200


@api.route("/ask", methods=["POST"])
def ask():
    """Ask Gemini a question and recall relevant audio memories"""
    data = request.get_json() or {}
    question = data.get("question", "")
    model = data.get("model", "gemini-2.5-flash")
    max_results = data.get("max_results", 5)
    memory_limit = data.get("memory_limit", 100)  # How many recent memories to search through
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        # Get recent memory nodes to search through
        all_memories = get_memory_nodes(limit=memory_limit)
        
        if not all_memories:
            return jsonify({
                "question": question,
                "memories": [],
                "message": "No memories found in database"
            }), 200
        
        # Use Gemini to find relevant memories
        relevant_memories = recall_memories(question, all_memories, model=model, max_results=max_results)
        
        # Format response with audio paths
        result = {
            "question": question,
            "memories": relevant_memories,
            "total_memories_searched": len(all_memories),
            "relevant_memories_found": len(relevant_memories)
        }
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to recall memories: {str(e)}"}), 500


# Speech recording endpoints
@api.route("/record/start", methods=["POST"])
def start_recording():
    """Start audio recording"""
    data = request.get_json() or {}
    # Generate unique filename if not provided
    if "output_path" not in data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"recordings/recording_{timestamp}.wav"
        # Ensure recordings directory exists
        os.makedirs("recordings", exist_ok=True)
    else:
        output_path = data.get("output_path")
    
    result, status_code = recorder.start_recording(output_path)
    return jsonify(result), status_code


@api.route("/record/stop", methods=["POST"])
def stop_recording():
    """Stop audio recording, transcribe, and save as memory node"""
    data = request.get_json() or {}
    video_path = data.get("video_path")  # Optional video path
    video_summary = data.get("video_summary")  # Optional video summary
    model = data.get("model", "gemini-2.5-flash")
    auto_transcribe = data.get("auto_transcribe", True)  # Default to True
    
    # Stop recording
    result, status_code = recorder.stop_recording()
    if status_code != 200:
        return jsonify(result), status_code
    
    # If auto_transcribe is enabled, transcribe and save as memory node
    if auto_transcribe and "output_path" in result:
        audio_path = result["output_path"]
        try:
            transcript, timestamp = transcribe_audio(audio_path, model)
            
            # Save to database as memory node
            node_id = add_memory_node(
                transcript, timestamp,
                audio_path=audio_path,
                video_path=video_path,
                video_summary=video_summary
            )
            
            # Add memory node info to result
            result["id"] = node_id
            result["transcript"] = transcript
            result["timestamp"] = timestamp
            
            if video_path:
                result["video_path"] = video_path
            if video_summary:
                result["video_summary"] = video_summary
                
        except Exception as e:
            # If transcription fails, still return the stop result but with error
            result["transcription_error"] = str(e)
    
    return jsonify(result), status_code


@api.route("/record/status", methods=["GET"])
def get_recording_status():
    """Get current recording status"""
    status = recorder.get_status()
    return jsonify(status), 200


# Transcription endpoints
@api.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe an audio file and save as memory node"""
    data = request.get_json() or {}
    audio_path = data.get("audio_path", "recording.wav")
    video_path = data.get("video_path")  # Optional video path
    video_summary = data.get("video_summary")  # Optional video summary
    model = data.get("model", "gemini-2.5-flash")
    save_to_file = data.get("save_to_file", False)  # Default to False, focus on DB
    
    try:
        transcript, timestamp = transcribe_audio(audio_path, model)
        
        # Save to database as memory node
        node_id = add_memory_node(
            transcript, timestamp, 
            audio_path=audio_path, 
            video_path=video_path,
            video_summary=video_summary
        )
        
        result = {
            "id": node_id,
            "transcript": transcript,
            "timestamp": timestamp,
            "audio_path": audio_path
        }
        
        if video_path:
            result["video_path"] = video_path
        
        if video_summary:
            result["video_summary"] = video_summary
        
        # Optionally save to file if requested
        if save_to_file:
            transcript_path = data.get("transcript_path", "transcript.txt")
            if save_transcript(transcript, timestamp, transcript_path):
                result["transcript_path"] = transcript_path
        
        return jsonify(result), 200
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500


@api.route("/transcript", methods=["GET"])
def get_transcript():
    """Get the latest transcript from file"""
    transcript_path = request.args.get("path", "transcript.txt")
    
    if not os.path.exists(transcript_path):
        return jsonify({"error": "Transcript file not found"}), 404
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse timestamp if present
        lines = content.split("\n", 2)
        timestamp = None
        transcript = content
        
        if len(lines) >= 2 and lines[0].startswith("Timestamp:"):
            timestamp = lines[0].replace("Timestamp:", "").strip()
            transcript = lines[2] if len(lines) > 2 else ""
        
        return jsonify({
            "transcript": transcript,
            "timestamp": timestamp,
            "path": transcript_path
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to read transcript: {str(e)}"}), 500


@api.route("/record-and-transcribe", methods=["POST"])
def record_and_transcribe():
    """Record audio and transcribe it in one request, saving as memory node"""
    data = request.get_json() or {}
    duration = data.get("duration", 5)  # Default 5 seconds
    output_path = data.get("output_path", "recording.wav")
    video_path = data.get("video_path")  # Optional video path
    video_summary = data.get("video_summary")  # Optional video summary
    model = data.get("model", "gemini-2.5-flash")
    save_to_file = data.get("save_to_file", False)  # Default to False, focus on DB
    
    import time
    
    # Start recording
    result, status_code = recorder.start_recording(output_path)
    if status_code != 200:
        return jsonify(result), status_code
    
    # Wait for specified duration
    time.sleep(duration)
    
    # Stop recording
    result, status_code = recorder.stop_recording()
    if status_code != 200:
        return jsonify(result), status_code
    
    # Transcribe
    try:
        transcript, timestamp = transcribe_audio(output_path, model)
        
        # Save to database as memory node
        node_id = add_memory_node(
            transcript, timestamp, 
            audio_path=output_path, 
            video_path=video_path,
            video_summary=video_summary
        )
        
        result = {
            "id": node_id,
            "transcript": transcript,
            "timestamp": timestamp,
            "audio_path": output_path,
            "duration": duration
        }
        
        if video_path:
            result["video_path"] = video_path
        
        if video_summary:
            result["video_summary"] = video_summary
        
        # Optionally save to file if requested
        if save_to_file:
            transcript_path = data.get("transcript_path", "transcript.txt")
            if save_transcript(transcript, timestamp, transcript_path):
                result["transcript_path"] = transcript_path
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500


# Audio playback endpoints
@api.route("/audio/<path:filename>", methods=["GET"])
def get_audio(filename):
    """Serve audio file by filename"""
    # Security: prevent directory traversal
    if ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
    
    # Try recordings directory first
    file_path = os.path.join("recordings", filename)
    if not os.path.exists(file_path):
        # Try current directory
        file_path = filename
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Audio file not found"}), 404
    
    return send_file(file_path, mimetype="audio/wav")


@api.route("/memories/<int:node_id>/audio", methods=["GET"])
def get_memory_audio(node_id):
    """Get audio file for a specific memory node"""
    memory = get_memory_node_by_id(node_id)
    if not memory:
        return jsonify({"error": "Memory node not found"}), 404
    
    audio_path = memory.get("audio_path")
    if not audio_path:
        return jsonify({"error": "No audio file associated with this memory"}), 404
    
    if not os.path.exists(audio_path):
        return jsonify({"error": "Audio file not found"}), 404
    
    return send_file(audio_path, mimetype="audio/wav")

