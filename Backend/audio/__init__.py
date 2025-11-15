"""
Audio module for recording and transcription
Can be used as a module or standalone CLI script
"""
from .audio_module import (
    SpeechRecorder,
    recorder,
    transcribe_audio,
    transcribe_with_gemini,
    save_transcript,
    get_api_key,
    record_to_wav,
    main,
    parse_args
)

__all__ = [
    "SpeechRecorder",
    "recorder",
    "transcribe_audio",
    "transcribe_with_gemini",
    "save_transcript",
    "get_api_key",
    "record_to_wav",
    "main",
    "parse_args"
]
