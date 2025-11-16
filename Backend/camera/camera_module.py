"""
Motion detection and enrichment pipeline for the BigBrother camera service.

This version is a simplified, single-function implementation focused on stability,
especially for macOS and Continuity Camera devices that may be unreliable.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple
import json

import cv2

# --- Dependency Imports & Fallbacks ---

try:  # Database helper
    from ..db.database import add_event, create_memory_node
except (ImportError, ModuleNotFoundError):
    add_event = None
    create_memory_node = None
    logging.warning("Database module not found. Events will be logged to console only.")


def _import_gemini_helpers() -> (Callable[[str], str], Callable[[str], str]):
    """Return `describe_image` and `summarize_video` callables if available, else safe fallbacks."""
    try:
        from ..ai.gemini_client import describe_image, summarize_video
        if not callable(describe_image):
            raise ImportError("describe_image not callable")
        if not callable(summarize_video):
            raise ImportError("summarize_video not callable")
        return describe_image, summarize_video
    except (ImportError, ModuleNotFoundError) as exc:
        logging.warning("Gemini client unavailable or incomplete: %s", exc)

    def _fallback_describe(image_path: str) -> str:
        return "Vision caption unavailable"
    
    def _fallback_summarize(video_path: str) -> str:
        return "Video summary unavailable"

    return _fallback_describe, _fallback_summarize


def _import_audio_helpers():
    """Return audio recorder and transcription functions if available, else None."""
    try:
        from ..audio.audio_module import recorder, transcribe_audio, save_transcript
        return recorder, transcribe_audio, save_transcript
    except (ImportError, ModuleNotFoundError) as exc:
        logging.warning("Audio module unavailable: %s", exc)
        return None, None, None


def _load_yolo_model():
    """Load YOLOv8 model if available, otherwise return None."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        logging.info("YOLOv8 model loaded successfully.")
        return model
    except (ImportError, ModuleNotFoundError, Exception) as exc:
        logging.warning("YOLO model unavailable: %s", exc)
        return None

def analyze_and_log_video(
    video_path: str,
    yolo_model,
    describe_image: Callable,
    summarize_video: Callable,
    image_dir: Path,
    audio_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    transcript: Optional[str] = None,
):
    """
    Analyzes a video in a separate thread, generates a summary, and logs the event.
    Creates a unified MemoryNode with video path, audio path, summary, and transcript.
    """
    try:
        logging.info(f"Starting analysis for {video_path}...")
        
        # For YOLO and initial description, we can still use the first frame.
        # This provides a quick preview/thumbnail.
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logging.error(f"Failed to read first frame from {video_path} for analysis.")
            return

        ts_utc = datetime.utcnow()
        first_frame_filename = f"thumbnail_{ts_utc.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        first_frame_path = image_dir / first_frame_filename
        cv2.imwrite(str(first_frame_path), frame)

        # --- AI Analysis ---
        # 1. YOLO object detection on the first frame
        objects = []
        if yolo_model:
            try:
                results = yolo_model(first_frame_path, verbose=False)
                if results and hasattr(results[0], "names"):
                    box_classes = results[0].boxes.cls.cpu().numpy()
                    objects = sorted(list(set(results[0].names[int(c)] for c in box_classes)))
            except Exception as e:
                logging.error(f"YOLO prediction failed: {e}")

        # 2. Gemini video summarization
        summary = ""
        try:
            summary = summarize_video(video_path)
        except Exception as e:
            logging.error(f"Gemini video summary failed: {e}")

        # --- Event Logging ---
        desc_parts = []
        if objects:
            desc_parts.append(f"Objects detected: {', '.join(objects)}")
        if summary:
            desc_parts.append(f"AI Summary: {summary}")
        
        description = " | ".join(desc_parts) or "Video recorded"
        
        logging.info(f"Analysis complete for {video_path}: {description}")

        if add_event:
            try:
                add_event(
                    event_type="video_recording",
                    timestamp=ts_utc.isoformat(),
                    description=description,
                    image_path=video_path, # The path to the video file
                )
            except Exception as e:
                logging.error(f"Failed to save video event to database: {e}")
        
        # Create or update unified MemoryNode with all data (video, audio, summary, transcript)
        if create_memory_node:
            try:
                from ..db.database import get_memory_node_by_file_path, update_memory_node_metadata
                
                # Check if MemoryNode already exists (might have been created by transcript thread)
                existing_node = get_memory_node_by_file_path(video_path)
                
                if existing_node:
                    # Update existing MemoryNode with video analysis data
                    try:
                        existing_metadata = json.loads(existing_node.get('metadata', '{}') or '{}')
                    except:
                        existing_metadata = {}
                    
                    # Update with video analysis data (summary, objects, etc.)
                    existing_metadata['video_path'] = video_path
                    existing_metadata['summary'] = summary
                    existing_metadata['objects_detected'] = objects
                    existing_metadata['description'] = description
                    existing_metadata['thumbnail_path'] = str(first_frame_path)
                    if audio_path:
                        existing_metadata['audio_path'] = audio_path
                    if transcript_path:
                        existing_metadata['transcript_path'] = transcript_path
                    if transcript:
                        existing_metadata['transcript'] = transcript
                    
                    # Update the node
                    if update_memory_node_metadata(existing_node['id'], existing_metadata):
                        logging.info(f"✓ Updated existing MemoryNode {existing_node['id']} with video analysis data")
                    else:
                        logging.error(f"✗ Failed to update existing MemoryNode {existing_node['id']}")
                else:
                    # Create new MemoryNode with all available data
                    metadata = {
                        "video_path": video_path,
                        "audio_path": audio_path,
                        "transcript_path": transcript_path,
                        "summary": summary,
                        "transcript": transcript,
                        "objects_detected": objects,
                        "description": description,
                        "thumbnail_path": str(first_frame_path)
                    }
                    create_memory_node(
                        file_path=video_path,  # Use video path as primary file path
                        file_type="recording",   # Changed to "recording" to indicate it's a complete recording session
                        timestamp=ts_utc.isoformat(),
                        metadata=json.dumps(metadata)
                    )
                    logging.info(f"✓ Created new MemoryNode for recording: {video_path}")
            except Exception as e:
                logging.error(f"Failed to create/update MemoryNode: {e}", exc_info=True)

    except Exception as e:
        logging.critical(f"An error occurred during video analysis for {video_path}: {e}", exc_info=True)


# --- Main Camera Loop ---

def run_camera_loop(
    camera_index: int,
    processing_fps: float,
    min_contour_area: int,
    image_dir: Path,
    stop_event: threading.Event,
    capture_width: Optional[int] = None,
    capture_height: Optional[int] = None,
    capture_fps: Optional[float] = None,
    max_frame_failures: int = 10,
    warmup_period: float = 2.0,
    inactivity_timeout: float = 5.0,
    delta_thresh: int = 50,
):
    """
    Main webcam loop for motion detection and event creation.
    """
    yolo_model = _load_yolo_model()
    describe_image, summarize_video = _import_gemini_helpers()
    audio_recorder, transcribe_audio, save_transcript = _import_audio_helpers()
    
    # Use a deque to store recent motion levels
    motion_history_length = int(processing_fps * 2)  # Store 2 seconds of motion data
    motion_history: Deque[float] = deque(maxlen=motion_history_length)
    
    image_dir.mkdir(parents=True, exist_ok=True)
    # Create a directory for video recordings
    recording_dir = image_dir.parent / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)
    # Create directories for audio recordings and transcripts
    audio_dir = image_dir.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = image_dir.parent / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    def _open_capture() -> cv2.VideoCapture:
        """Opens, configures, and primes the video capture device."""
        # Use CAP_AVFOUNDATION for better macOS compatibility
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(f"FATAL: Unable to open webcam at index {camera_index}")

        # Apply requested settings
        if capture_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        if capture_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        if capture_fps:
            cap.set(cv2.CAP_PROP_FPS, capture_fps)
        
        # Log the actual negotiated settings
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Camera opened: {camera_index} @ {actual_w}x{actual_h}, {actual_fps:.2f} FPS")
        
        # --- Priming Loop ---
        logging.info("Priming camera stream...")
        for i in range(30): # Try for ~3 seconds
            ret, _ = cap.read()
            if ret:
                logging.info(f"Stream is live after {i + 1} attempts.")
                return cap
            time.sleep(0.1)

        # If priming fails, release the resource and raise an error.
        cap.release()
        raise RuntimeError("FATAL: Camera opened but failed to start streaming.")

    cap = _open_capture()
    is_recording = False
    last_motion_time = None
    video_writer = None
    consecutive_failures = 0
    audio_path = None  # Track current audio recording path
    current_timestamp_str = None  # Track timestamp for current recording
    
    # We need two frames for comparison
    prev_gray_frame = None
    
    logging.info(f"Starting motion detection loop at ~{processing_fps:.1f} FPS.")

    try:
        while not stop_event.is_set():
            read_start_time = time.monotonic()
            
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                logging.warning(f"Frame grab failed ({consecutive_failures}/{max_frame_failures})")
                if consecutive_failures >= max_frame_failures:
                    logging.error("Exceeded max frame failures. Aborting.")
                    break
                # Attempt to recover by re-opening the capture device
                cap.release()
                try:
                    cap = _open_capture()
                    consecutive_failures = 0 # Reset on successful reopen
                except RuntimeError as e:
                    logging.error(f"Failed to reopen camera: {e}. Retrying in 5s.")
                    time.sleep(5)
                continue

            # --- Motion Detection Logic ---
            small_frame = cv2.resize(frame, (640, 480))
            gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            if prev_gray_frame is None:
                prev_gray_frame = gray_frame
                continue

            delta = cv2.absdiff(prev_gray_frame, gray_frame)
            thresh = cv2.threshold(delta, delta_thresh, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Update the previous frame for the next iteration
            prev_gray_frame = gray_frame

            # Calculate a "motion level" for the current frame
            total_contour_area = sum(cv2.contourArea(c) for c in contours)
            motion_history.append(total_contour_area)
            
            # Determine if significant motion is happening based on history
            # We consider motion to be happening if the average contour area is above our min threshold
            avg_motion = sum(motion_history) / len(motion_history) if motion_history else 0
            motion_detected = avg_motion > min_contour_area

            if motion_detected:
                last_motion_time = time.monotonic()
                if not is_recording:
                    # --- Start of a new recording ---
                    is_recording = True
                    ts_utc = datetime.utcnow()
                    current_timestamp_str = ts_utc.strftime('%Y%m%d_%H%M%S')
                    video_filename = f"motion_{current_timestamp_str}.mp4"
                    video_path = str(recording_dir / video_filename)
                    
                    # Get frame dimensions for VideoWriter
                    frame_height, frame_width, _ = frame.shape
                    
                    # Define the codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, capture_fps or processing_fps, (frame_width, frame_height))
                    
                    # Start audio recording
                    if audio_recorder:
                        audio_filename = f"motion_{current_timestamp_str}.wav"
                        audio_path = str(audio_dir / audio_filename)
                        result, status_code = audio_recorder.start_recording(audio_path)
                        if status_code == 200:
                            logging.info(f"Audio recording started: {audio_path}")
                        else:
                            logging.warning(f"Failed to start audio recording: {result.get('error', 'Unknown error')}")
                            audio_path = None
                    
                    logging.info(f"Motion detected (avg level: {avg_motion:.0f})! Starting new recording: {video_path}")
            
            if is_recording:
                if video_writer:
                    video_writer.write(frame)

                # Check for inactivity to stop recording
                # Inactivity is now defined as a lack of significant motion for the timeout period
                if time.monotonic() - (last_motion_time or 0) > inactivity_timeout:
                    logging.info(f"Motion level below threshold for {inactivity_timeout}s. Stopping recording: {video_path}")
                    is_recording = False
                    
                    # Stop audio recording
                    current_audio_path = audio_path
                    current_video_path = video_path  # Capture video_path before releasing
                    timestamp_for_transcript = current_timestamp_str if current_timestamp_str else datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                    if audio_recorder and current_audio_path:
                        result, status_code = audio_recorder.stop_recording()
                        if status_code == 200:
                            logging.info(f"Audio recording stopped: {current_audio_path}")
                            
                            # Transcribe audio in a separate thread
                            def transcribe_and_save():
                                try:
                                    logging.info(f"Starting transcription for: {current_audio_path}")
                                    transcript, timestamp = transcribe_audio(current_audio_path)
                                    
                                    # Save transcript to file with matching timestamp
                                    transcript_filename = f"motion_{timestamp_for_transcript}.txt"
                                    transcript_path = str(transcript_dir / transcript_filename)
                                    transcript_saved = save_transcript(transcript, timestamp, transcript_path)
                                    
                                    if transcript_saved:
                                        logging.info(f"Transcript saved: {transcript_path}")
                                    else:
                                        logging.error(f"Failed to save transcript: {transcript_path}")
                                    
                                    # Update or create MemoryNode with transcript data
                                    # We need to either update existing MemoryNode or create one if it doesn't exist yet
                                    if create_memory_node and current_video_path and transcript:
                                        try:
                                            from ..db.database import (
                                                get_memory_node_by_file_path, 
                                                update_memory_node_metadata,
                                                create_memory_node as create_node
                                            )
                                            
                                            # Wait for MemoryNode to be created (with retries)
                                            max_retries = 15  # More retries since video analysis can take time
                                            retry_delay = 2.0  # 2 seconds between retries
                                            video_node = None
                                            
                                            logging.info(f"Looking for MemoryNode to update with transcript: {current_video_path}")
                                            
                                            for attempt in range(max_retries):
                                                video_node = get_memory_node_by_file_path(current_video_path)
                                                if video_node:
                                                    logging.info(f"✓ Found MemoryNode {video_node['id']} on attempt {attempt + 1}")
                                                    break
                                                if attempt < max_retries - 1:
                                                    logging.debug(f"MemoryNode not found yet, waiting {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                                                    time.sleep(retry_delay)
                                            
                                            if video_node:
                                                # Get existing metadata
                                                try:
                                                    existing_metadata = json.loads(video_node.get('metadata', '{}') or '{}')
                                                except:
                                                    existing_metadata = {}
                                                
                                                # Update with transcript data
                                                existing_metadata['transcript'] = transcript
                                                existing_metadata['transcript_path'] = transcript_path
                                                existing_metadata['audio_path'] = current_audio_path
                                                
                                                # Update the node
                                                if update_memory_node_metadata(video_node['id'], existing_metadata):
                                                    logging.info(f"✓ Successfully updated MemoryNode {video_node['id']} with transcript ({len(transcript)} characters)")
                                                    
                                                    # Verify the update worked
                                                    time.sleep(0.5)  # Brief pause before verification
                                                    updated_node = get_memory_node_by_file_path(current_video_path)
                                                    if updated_node:
                                                        try:
                                                            verify_metadata = json.loads(updated_node.get('metadata', '{}') or '{}')
                                                            if verify_metadata.get('transcript'):
                                                                logging.info(f"✓✓ Verified: transcript is stored in MemoryNode {video_node['id']}")
                                                                logging.info(f"   Transcript preview: {transcript[:50]}...")
                                                            else:
                                                                logging.error(f"✗ Verification failed: transcript not found in MemoryNode {video_node['id']}")
                                                                logging.error(f"   Metadata keys: {list(verify_metadata.keys())}")
                                                        except Exception as e:
                                                            logging.error(f"✗ Error verifying metadata: {e}")
                                                else:
                                                    logging.error(f"✗ Failed to update MemoryNode {video_node['id']} with transcript")
                                            else:
                                                # MemoryNode doesn't exist yet - create it with transcript
                                                logging.warning(f"⚠ MemoryNode not found after {max_retries} attempts. Creating new one with transcript.")
                                                try:
                                                    # Create a basic MemoryNode with transcript
                                                    # The video summary will be added later by analyze_and_log_video
                                                    metadata = {
                                                        "video_path": current_video_path,
                                                        "audio_path": current_audio_path,
                                                        "transcript_path": transcript_path,
                                                        "summary": None,  # Will be updated later
                                                        "transcript": transcript,
                                                        "objects_detected": [],
                                                        "description": "Recording with transcript",
                                                    }
                                                    node_id = create_node(
                                                        file_path=current_video_path,
                                                        file_type="recording",
                                                        timestamp=timestamp,
                                                        metadata=json.dumps(metadata)
                                                    )
                                                    logging.info(f"✓ Created MemoryNode {node_id} with transcript ({len(transcript)} characters)")
                                                except Exception as e:
                                                    logging.error(f"✗ Failed to create MemoryNode with transcript: {e}", exc_info=True)
                                        except Exception as e:
                                            logging.error(f"✗ Error handling transcript in MemoryNode: {e}", exc_info=True)
                                    elif not transcript:
                                        logging.warning(f"⚠ No transcript to store for {current_video_path}")
                                    
                                except Exception as e:
                                    logging.error(f"Error transcribing audio {current_audio_path}: {e}", exc_info=True)
                            
                            transcription_thread = threading.Thread(target=transcribe_and_save, daemon=True)
                            transcription_thread.start()
                        else:
                            logging.warning(f"Failed to stop audio recording: {result.get('error', 'Unknown error')}")
                    
                    if video_writer:
                        video_writer.release()
                        
                        # Start analysis in a new thread
                        # Pass audio_path and transcript_path so we can create unified MemoryNode
                        analysis_thread = threading.Thread(
                            target=analyze_and_log_video,
                            args=(
                                current_video_path,
                                yolo_model,
                                describe_image,
                                summarize_video,
                                image_dir,
                                current_audio_path,  # audio_path
                                None,  # transcript_path (will be set later)
                                None,  # transcript (will be set later)
                            ),
                            daemon=True,
                        )
                        analysis_thread.start()

                        video_writer = None
                    audio_path = None
                    current_timestamp_str = None
            
            # --- Frame Rate Control ---
            elapsed = time.monotonic() - read_start_time
            sleep_time = (1.0 / processing_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # Stop audio recording if still active
        if audio_recorder and audio_path:
            try:
                audio_recorder.stop_recording()
                logging.info("Audio recording stopped on exit")
            except Exception as e:
                logging.warning(f"Error stopping audio recording on exit: {e}")
        
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if video_writer: # Ensure writer is released on exit
            video_writer.release()
        logging.info("Camera loop stopped.")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Run motion detection camera loop.")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the webcam.")
    parser.add_argument("--fps", type=float, default=10.0, help="Processing frames per second.")
    parser.add_argument("--min-area", type=int, default=2000, help="Minimum contour area to trigger motion.")
    parser.add_argument("--width", type=int, help="Requested capture width (pixels).")
    parser.add_argument("--height", type=int, help="Requested capture height (pixels).")
    parser.add_argument("--capture-fps", type=float, help="Requested capture FPS from the device.")
    parser.add_argument("--max-frame-failures", type=int, default=10, help="Max consecutive frame read failures.")
    parser.add_argument("--inactivity-timeout", type=float, default=5.0, help="Seconds of no motion to stop recording.")
    parser.add_argument("--delta-thresh", type=int, default=50, help="Threshold for detecting pixel changes (1-255).")
    args = parser.parse_args(argv)

    stop_event = threading.Event()
    
    # Set up a thread to listen for Ctrl+C
    def signal_handler():
        try:
            input("Press Enter or Ctrl+C to stop...\n")
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            logging.info("Stop signal received.")
            stop_event.set()

    signal_thread = threading.Thread(target=signal_handler, daemon=True)
    signal_thread.start()

    base_dir = Path(__file__).resolve().parents[1]
    image_dir = base_dir / "data" / "images"

    try:
        run_camera_loop(
            camera_index=args.camera_index,
            processing_fps=args.fps,
            min_contour_area=args.min_area,
            image_dir=image_dir,
            stop_event=stop_event,
            capture_width=args.width,
            capture_height=args.height,
            capture_fps=args.capture_fps,
            max_frame_failures=args.max_frame_failures,
            inactivity_timeout=args.inactivity_timeout,
            delta_thresh=args.delta_thresh,
        )
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
