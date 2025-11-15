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
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

# --- Dependency Imports & Fallbacks ---

try:  # Database helper
    from ..db.database import add_event
except (ImportError, ModuleNotFoundError):
    add_event = None
    logging.warning("Database module not found. Events will be logged to console only.")


def _import_gemini_describer() -> Callable[[str], str]:
    """Return `describe_image` callable if available, else a safe fallback."""
    try:
        from ..ai.gemini_client import describe_image
        if callable(describe_image):
            return describe_image
    except (ImportError, ModuleNotFoundError) as exc:
        logging.warning("Gemini client unavailable: %s", exc)

    def _fallback(image_path: str) -> str:
        return "Vision caption unavailable"
    return _fallback


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
):
    """
    Main webcam loop for motion detection and event creation.
    """
    yolo_model = _load_yolo_model()
    describe_image = _import_gemini_describer()
    ref_frame = None
    
    image_dir.mkdir(parents=True, exist_ok=True)

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
    consecutive_failures = 0
    
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

            if ref_frame is None:
                ref_frame = gray_frame
                continue

            delta = cv2.absdiff(ref_frame, gray_frame)
            thresh = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = any(cv2.contourArea(c) >= min_contour_area for c in contours)

            if motion_detected:
                logging.info("Motion detected!")
                ref_frame = gray_frame  # Update reference to avoid re-triggering

                # --- Enrichment and Storage ---
                ts_utc = datetime.utcnow()
                filename = f"motion_{ts_utc.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                image_path = image_dir / filename
                cv2.imwrite(str(image_path), frame)

                # YOLO object detection
                objects = []
                if yolo_model:
                    try:
                        results = yolo_model(image_path, verbose=False)
                        if results and hasattr(results[0], "names"):
                            box_classes = results[0].boxes.cls.cpu().numpy()
                            objects = sorted(list(set(results[0].names[int(c)] for c in box_classes)))
                    except Exception as e:
                        logging.error(f"YOLO prediction failed: {e}")

                # Gemini captioning
                caption = ""
                try:
                    caption = describe_image(str(image_path))
                except Exception as e:
                    logging.error(f"Gemini caption failed: {e}")

                # Build description and save event
                desc_parts = []
                if objects:
                    desc_parts.append(f"Objects: {', '.join(objects)}")
                if caption:
                    desc_parts.append(f"Caption: {caption}")
                description = " | ".join(desc_parts) or "Motion detected"
                
                logging.info(f"Event: {description}")

                if add_event:
                    try:
                        add_event(
                            event_type="motion",
                            timestamp=ts_utc.isoformat(),
                            description=description,
                            image_path=str(image_path),
                        )
                    except Exception as e:
                        logging.error(f"Failed to save event to database: {e}")

            # --- Frame Rate Control ---
            elapsed = time.monotonic() - read_start_time
            sleep_time = (1.0 / processing_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        if cap.isOpened():
            cap.release()
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
        )
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
