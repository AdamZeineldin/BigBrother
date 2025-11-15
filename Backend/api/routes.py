from flask import Blueprint, request, jsonify
import os

from db.database import get_events, search_events
from audio import recorder, transcribe_audio, save_transcript

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
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

