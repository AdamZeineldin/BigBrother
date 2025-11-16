from flask import Blueprint, request, jsonify
import os
import json

from db.database import (
    get_events, 
    search_events, 
    create_memory_node, 
    get_all_memory_nodes_for_search,
    get_memory_nodes,
    get_memory_node_by_id,
    cleanup_orphaned_memory_nodes
)
from audio import recorder, transcribe_audio, save_transcript
from ai.gemini_client import search_memory_nodes as gemini_search_memory_nodes
from camera.camera_service import get_camera_service

api = Blueprint("api", __name__)

@api.route("/events")
def events():
    return {"events": get_events()}

@api.route("/search")
def search():
    q = request.args.get("q", "")
    return {"events": search_events(q)}


# Speech recording endpoints
@api.route("/record/start", methods=["POST"])
def start_recording():
    """Start audio recording"""
    data = request.get_json() or {}
    output_path = data.get("output_path", "recording.wav")
    
    result, status_code = recorder.start_recording(output_path)
    return jsonify(result), status_code


@api.route("/record/stop", methods=["POST"])
def stop_recording():
    """Stop audio recording"""
    result, status_code = recorder.stop_recording()
    return jsonify(result), status_code


@api.route("/record/status", methods=["GET"])
def get_recording_status():
    """Get current recording status"""
    status = recorder.get_status()
    return jsonify(status), 200


# Transcription endpoints
@api.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe an audio file"""
    data = request.get_json() or {}
    audio_path = data.get("audio_path", "recording.wav")
    model = data.get("model", "gemini-2.5-flash")
    save_to_file = data.get("save_to_file", True)
    
    try:
        transcript, timestamp = transcribe_audio(audio_path, model)
        
        result = {
            "transcript": transcript,
            "timestamp": timestamp,
            "audio_path": audio_path
        }
        
        if save_to_file:
            transcript_path = data.get("transcript_path", "transcript.txt")
            if save_transcript(transcript, timestamp, transcript_path):
                result["transcript_path"] = transcript_path
                
                # Create MemoryNode for transcript (unified structure)
                try:
                    metadata = {
                        "video_path": None,  # No video for API-only transcripts
                        "audio_path": audio_path,
                        "transcript_path": transcript_path,
                        "summary": None,  # No video summary for audio-only
                        "transcript": transcript,
                        "objects_detected": [],
                        "description": "Audio transcription"
                    }
                    create_memory_node(
                        file_path=transcript_path,  # Use transcript path as primary
                        file_type="recording",  # Use "recording" for consistency
                        timestamp=timestamp,
                        metadata=json.dumps(metadata)
                    )
                except Exception as e:
                    # Log error but don't fail the request
                    print(f"Failed to create MemoryNode for transcript: {e}")
        
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
    """Record audio and transcribe it in one request"""
    data = request.get_json() or {}
    duration = data.get("duration", 5)  # Default 5 seconds
    output_path = data.get("output_path", "recording.wav")
    model = data.get("model", "gemini-2.5-flash")
    
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
        
        result = {
            "transcript": transcript,
            "timestamp": timestamp,
            "audio_path": output_path,
            "duration": duration
        }
        
        # Save transcript
        transcript_path = data.get("transcript_path", "transcript.txt")
        if save_transcript(transcript, timestamp, transcript_path):
            result["transcript_path"] = transcript_path
            
            # Create MemoryNode for transcript (unified structure)
            try:
                metadata = {
                    "video_path": None,  # No video for API-only transcripts
                    "audio_path": output_path,
                    "transcript_path": transcript_path,
                    "summary": None,  # No video summary for audio-only
                    "transcript": transcript,
                    "objects_detected": [],
                    "description": "Audio transcription"
                }
                create_memory_node(
                    file_path=transcript_path,  # Use transcript path as primary
                    file_type="recording",  # Use "recording" for consistency
                    timestamp=timestamp,
                    metadata=json.dumps(metadata)
                )
            except Exception as e:
                # Log error but don't fail the request
                print(f"Failed to create MemoryNode for transcript: {e}")
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500


# MemoryNode endpoints
@api.route("/memory-nodes", methods=["GET"])
def get_memory_nodes_endpoint():
    """Get all MemoryNodes, optionally filtered by file_type"""
    file_type = request.args.get("file_type")  # 'recording' (unified) or legacy types
    limit = request.args.get("limit", type=int)
    
    try:
        nodes = get_memory_nodes(file_type=file_type, limit=limit)
        return jsonify({"memory_nodes": nodes}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to get MemoryNodes: {str(e)}"}), 500


@api.route("/memory-nodes/<int:node_id>", methods=["GET"])
def get_memory_node_endpoint(node_id):
    """Get a specific MemoryNode by ID"""
    try:
        node = get_memory_node_by_id(node_id)
        if node:
            return jsonify(node), 200
        else:
            return jsonify({"error": "MemoryNode not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to get MemoryNode: {str(e)}"}), 500


@api.route("/memory-nodes/search", methods=["POST"])
def search_memory_nodes_endpoint():
    """Search MemoryNodes using Gemini AI"""
    data = request.get_json() or {}
    query = data.get("query", "")
    max_results = data.get("max_results", 5)
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        # Get all MemoryNodes for search
        all_nodes = get_all_memory_nodes_for_search()
        
        if not all_nodes:
            return jsonify({"memory_nodes": []}), 200
        
        # Use Gemini to search
        results = gemini_search_memory_nodes(
            query=query,
            memory_nodes=all_nodes,
            max_results=max_results
        )
        
        return jsonify({
            "query": query,
            "memory_nodes": results,
            "total_found": len(results)
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@api.route("/memory-nodes/cleanup", methods=["POST"])
def cleanup_orphaned_memory_nodes_endpoint():
    """Remove memory nodes whose associated files no longer exist"""
    try:
        deleted_count, deleted_ids = cleanup_orphaned_memory_nodes()
        return jsonify({
            "message": f"Cleaned up {deleted_count} orphaned memory nodes",
            "deleted_count": deleted_count,
            "deleted_ids": deleted_ids
        }), 200
    except Exception as e:
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500


# Camera service endpoints
@api.route("/camera/start", methods=["POST"])
def start_camera():
    """Start the camera module for motion detection and recording"""
    data = request.get_json() or {}
    
    try:
        camera_service = get_camera_service()
        result, status_code = camera_service.start(**data)
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({"error": f"Failed to start camera: {str(e)}"}), 500


@api.route("/camera/stop", methods=["POST"])
def stop_camera():
    """Stop the camera module"""
    try:
        camera_service = get_camera_service()
        result, status_code = camera_service.stop()
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({"error": f"Failed to stop camera: {str(e)}"}), 500


@api.route("/camera/status", methods=["GET"])
def get_camera_status():
    """Get the current camera service status"""
    try:
        camera_service = get_camera_service()
        status = camera_service.get_status()
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"error": f"Failed to get camera status: {str(e)}"}), 500

