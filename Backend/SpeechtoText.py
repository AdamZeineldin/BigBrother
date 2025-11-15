import os
import sys
import argparse
import queue
import sounddevice as sd
import soundfile as sf
from datetime import datetime  # <-- added

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  



SAMPLE_RATE = 16_000
CHANNELS = 1
SUBTYPE = "PCM_16"

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def record_to_wav(output_path: str):
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


def get_api_key():
    """
    Load API key from the .env file variable: GOOGLE_API_KEY
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


def parse_args():
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

    # ---- Save transcript to file (Option A) with timestamp ----
    output_path = "transcript.txt"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {run_timestamp}\n\n")
            f.write(transcript)
        print(f"\nTranscript written to {output_path}")
    except OSError as e:
        print(f"Failed to write transcript to file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
