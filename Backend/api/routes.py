from flask import Blueprint, request, jsonify, send_from_directory, Response
import os
import json
import requests
import logging

LOGGER = logging.getLogger(__name__)

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
from ai.gemini_client import search_memory_nodes as gemini_search_memory_nodes, generate_short_answer
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


@api.route("/files/<path:filepath>")
def serve_file(filepath):
    """Serve files from the data directory (videos, audio, transcripts, images)"""
    try:
        # Get the backend directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(backend_dir, "data")
        file_path = os.path.join(data_dir, filepath)
        
        # Security check: ensure the file is within the data directory
        data_dir_abs = os.path.abspath(data_dir)
        file_path_abs = os.path.abspath(file_path)
        
        if not file_path_abs.startswith(data_dir_abs):
            return jsonify({"error": "Access denied"}), 403
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Get the directory and filename
        file_dir = os.path.dirname(file_path_abs)
        filename = os.path.basename(file_path_abs)
        
        return send_from_directory(file_dir, filename)
    except Exception as e:
        return jsonify({"error": f"Failed to serve file: {str(e)}"}), 500


@api.route("/generate-answer-audio", methods=["POST"])
def generate_answer_audio():
    """Generate a short answer using Gemini and convert it to speech using ElevenLabs"""
    data = request.get_json() or {}
    query = data.get("query", "")
    summary = data.get("summary", "")
    video_path = data.get("video_path", None)
    audio_path = data.get("audio_path", None)
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    if not summary:
        return jsonify({"error": "Summary parameter is required"}), 400
    
    try:
        # Resolve full paths if relative paths are provided
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(backend_dir, "data")
        
        resolved_video_path = None
        if video_path:
            # Check if it's already an absolute path
            if os.path.isabs(video_path):
                resolved_video_path = video_path if os.path.exists(video_path) else None
            else:
                # Try to resolve relative to data directory
                if "recordings/" in video_path or video_path.startswith("recordings/"):
                    # Extract filename
                    filename = os.path.basename(video_path) if "/" in video_path else video_path
                    full_path = os.path.join(data_dir, "recordings", filename)
                    resolved_video_path = full_path if os.path.exists(full_path) else None
                else:
                    # Try as-is in data directory
                    full_path = os.path.join(data_dir, video_path)
                    resolved_video_path = full_path if os.path.exists(full_path) else None
        
        resolved_audio_path = None
        if audio_path:
            # Check if it's already an absolute path
            if os.path.isabs(audio_path):
                resolved_audio_path = audio_path if os.path.exists(audio_path) else None
            else:
                # Try to resolve relative to data directory
                if "audio/" in audio_path or audio_path.startswith("audio/"):
                    # Extract filename
                    filename = os.path.basename(audio_path) if "/" in audio_path else audio_path
                    full_path = os.path.join(data_dir, "audio", filename)
                    resolved_audio_path = full_path if os.path.exists(full_path) else None
                else:
                    # Try as-is in data directory
                    full_path = os.path.join(data_dir, audio_path)
                    resolved_audio_path = full_path if os.path.exists(full_path) else None
        
        # Step 1: Generate short answer using Gemini with video and audio
        answer = generate_short_answer(
            query=query, 
            summary=summary,
            video_path=resolved_video_path,
            audio_path=resolved_audio_path
        )
        
        if not answer:
            return jsonify({"error": "Failed to generate answer"}), 500
        
        # Step 2: Convert answer to speech using ElevenLabs
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not elevenlabs_api_key:
            # If no ElevenLabs API key, return the text answer
            return jsonify({
                "answer": answer,
                "audio_url": None,
                "error": "ELEVENLABS_API_KEY not configured"
            }), 200
        
        # ElevenLabs API endpoint
        elevenlabs_url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        payload = {
            "text": answer,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        # Make request to ElevenLabs
        response = requests.post(elevenlabs_url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Return the audio as a response
            return Response(
                response.content,
                mimetype="audio/mpeg",
                headers={
                    "Content-Disposition": "inline; filename=answer.mp3",
                    "X-Answer-Text": answer  # Include answer text in header for frontend
                }
            )
        else:
            # If ElevenLabs fails, return the text answer
            LOGGER.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return jsonify({
                "answer": answer,
                "audio_url": None,
                "error": f"ElevenLabs API error: {response.status_code}"
            }), 200
            
    except Exception as e:
        LOGGER.error(f"Failed to generate answer audio: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate answer audio: {str(e)}"}), 500


@api.route("/save-event", methods=["POST"])
def save_event_to_json():
    """Save event data to target.json file in the src directory"""
    data = request.get_json() or {}
    
    # Extract filename from path (just the basename)
    def get_filename(file_path):
        if not file_path:
            return None
        return os.path.basename(file_path)
    
    # Extract the required fields and file names
    event_data = {
        "title": data.get("title"),
        "timestamp": data.get("timestamp"),
        "summary": data.get("summary"),
        "transcript": data.get("transcript"),
        "recording": get_filename(data.get("video_path")),
        "audio": get_filename(data.get("audio_path")),
        "transcript_file": get_filename(data.get("transcript_path")),
        "image": get_filename(data.get("thumbnail_path")),
    }
    
    try:
        # Path to target.json in src directory
        # Backend is at project_root/Backend, so target.json is at project_root/src/target.json
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(backend_dir)
        target_json_path = os.path.join(project_root, "src", "target.json")
        
        # Write JSON data to file
        with open(target_json_path, "w", encoding="utf-8") as f:
            json.dump(event_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            "message": "Event data saved to target.json",
            "path": target_json_path
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to save event data: {str(e)}"}), 500


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

