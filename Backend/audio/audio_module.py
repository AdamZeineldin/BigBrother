"""
Audio module for Flask integration
Handles audio recording and transcription using Gemini API
Can also be used as a standalone CLI script (like SpeechtoText.py)
"""
# Standard library imports
import argparse
import os
import queue
import sys
import threading
from datetime import datetime

# Third-party imports
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SAMPLE_RATE = 16_000
CHANNELS = 1
SUBTYPE = "PCM_16"

# Global queue for standalone recording (for CLI compatibility)
audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Callback for audio input stream (for standalone CLI usage)"""
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def record_to_wav(output_path: str):
    """
    Record audio to WAV file (blocking, until Ctrl+C)
    This matches the original SpeechtoText.py behavior
    """
    print(f"Recording to {output_path}")
    print("Press Ctrl+C to stop.\n")

    with sf.SoundFile(
        output_path,
        mode="w",
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        subtype=SUBTYPE,
    ) as wav_file:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=audio_callback,
        ):
            try:
                while True:
                    wav_file.write(audio_queue.get())
            except KeyboardInterrupt:
                print("\nRecording stopped.")


class SpeechRecorder:
    """Thread-safe audio recorder for Flask"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        self.wav_file = None
        self.stream = None
        self.output_path = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if status:
            print(f"[Audio status] {status}", file=sys.stderr)
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def start_recording(self, output_path: str = "recording.wav"):
        """Start recording audio to a WAV file"""
        if self.is_recording:
            return {"error": "Recording already in progress"}, 400
        
        self.output_path = output_path
        self.is_recording = True
        self.audio_queue = queue.Queue()
        
        try:
            self.wav_file = sf.SoundFile(
                output_path,
                mode="w",
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                subtype=SUBTYPE,
            )
            
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self.audio_callback,
            )
            
            self.stream.start()
            
            # Start thread to write audio data
            self.recording_thread = threading.Thread(target=self._write_audio_data)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            return {"message": "Recording started", "output_path": output_path}, 200
            
        except Exception as e:
            self.is_recording = False
            return {"error": f"Failed to start recording: {str(e)}"}, 500
    
    def _write_audio_data(self):
        """Write audio data from queue to file"""
        try:
            while self.is_recording or not self.audio_queue.empty():
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    if self.wav_file:
                        self.wav_file.write(data)
                except queue.Empty:
                    if not self.is_recording:
                        # If recording stopped and queue is empty, exit
                        break
                    continue
        except Exception as e:
            print(f"Error writing audio data: {e}", file=sys.stderr)
    
    def stop_recording(self):
        """Stop recording and close file/stream"""
        if not self.is_recording:
            return {"error": "No recording in progress"}, 400
        
        self.is_recording = False
        
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
            
            if self.wav_file:
                self.wav_file.close()
            
            if self.recording_thread:
                self.recording_thread.join(timeout=2.0)
            
            return {
                "message": "Recording stopped",
                "output_path": self.output_path
            }, 200
            
        except Exception as e:
            return {"error": f"Failed to stop recording: {str(e)}"}, 500
    
    def get_status(self):
        """Get current recording status"""
        return {
            "is_recording": self.is_recording,
            "output_path": self.output_path if self.is_recording else None
        }


def get_api_key():
    """
    Load API key from the .env file variable: GOOGLE_API_KEY
    Matches original SpeechtoText.py behavior (exits on error for CLI)
    """
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print(
            "ERROR: GOOGLE_API_KEY not found.\n"
            "Make sure your .env file contains:\n\n"
            "    GOOGLE_API_KEY=your_key_here\n\n",
            file=sys.stderr
        )
        sys.exit(1)
    return key


def transcribe_with_gemini(audio_path: str, model: str = "gemini-2.5-flash"):
    """
    Transcribe audio file using Gemini API
    Matches original SpeechtoText.py signature (returns just transcript string)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    api_key = get_api_key()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            "Transcribe the speech:",
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav",
            ),
        ],
    )

    return (response.text or "").strip()


def transcribe_audio(audio_path: str, model: str = "gemini-2.5-flash"):
    """
    Transcribe audio file using Gemini API
    Returns tuple of (transcript_text, timestamp) for Flask usage
    """
    transcript = transcribe_with_gemini(audio_path, model)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return transcript, timestamp


def save_transcript(transcript: str, timestamp: str, output_path: str = "transcript.txt"):
    """Save transcript to file with timestamp"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(transcript)
        return True
    except OSError as e:
        print(f"Failed to write transcript to file: {e}", file=sys.stderr)
        return False


def parse_args():
    """Parse command line arguments (for CLI usage)"""
    parser = argparse.ArgumentParser(description="Gemini Audio Transcriber")
    parser.add_argument("-o", "--output", default="recording.wav")
    parser.add_argument(
        "-f",
        "--file",
        help="Use an existing audio file instead of recording",
    )
    parser.add_argument(
        "-m", "--model", default="gemini-2.5-flash"
    )
    return parser.parse_args()


def main():
    """
    Main function for CLI usage
    Matches original SpeechtoText.py behavior exactly
    """
    # Capture timestamp when the script starts processing this run
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    args = parse_args()

    if args.file:
        audio_path = args.file
    else:
        audio_path = args.output
        record_to_wav(audio_path)

    print("\nTranscribing... (this may take a moment)\n")

    try:
        transcript = transcribe_with_gemini(audio_path, model=args.model)
    except Exception as e:
        print(f"Transcription error: {e}", file=sys.stderr)
        sys.exit(1)

    print("=== TRANSCRIPT ===")
    print(transcript)
    print("==================")

    # Save transcript to file with timestamp
    output_path = "transcript.txt"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {run_timestamp}\n\n")
            f.write(transcript)
        print(f"\nTranscript written to {output_path}")
    except OSError as e:
        print(f"Failed to write transcript to file: {e}", file=sys.stderr)


# Global recorder instance for Flask
recorder = SpeechRecorder()


if __name__ == "__main__":
    main()
