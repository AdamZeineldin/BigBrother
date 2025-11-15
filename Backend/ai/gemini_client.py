"""Gemini 2.5 Flash image captioning helper."""

from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional helper
    load_dotenv = None  # type: ignore

try:
    import google.generativeai as genai
except Exception as exc:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_PROMPT = (
    "Provide a concise natural-language caption that describes the key objects, "
    "people, and actions that appear in this image."
)

if load_dotenv:
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

_MODEL = None

__all__ = ["describe_image"]


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
    return api_key


def _get_model():
    global _MODEL

    if genai is None:
        raise RuntimeError(
            "google-generativeai is not available. Install the optional dependency and try again."
        ) from _IMPORT_ERROR

    if _MODEL is None:
        api_key = _get_api_key()
        genai.configure(api_key=api_key)
        model_name = DEFAULT_MODEL_NAME
        _MODEL = genai.GenerativeModel(model_name)
        LOGGER.info("Initialized Gemini model: %s", model_name)

    return _MODEL


def describe_image(image_path: str, prompt: Optional[str] = None, timeout: int = 60) -> str:
    """Return a natural-language caption produced by Gemini 2.5 Flash.

    Args:
        image_path: Path to a local image file.
        prompt: Optional custom instruction. Defaults to DEFAULT_PROMPT.
        timeout: Seconds to wait for Gemini response.

    Raises:
        FileNotFoundError: if the image path does not exist.
        RuntimeError: if configuration or Gemini API calls fail.

    Returns:
        Caption text string. Falls back to "No caption returned" if model returns empty content.
    """

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = _get_model()
    instruction = prompt or DEFAULT_PROMPT

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    with path.open("rb") as fh:
        image_bytes = fh.read()

    try:
        response = model.generate_content(
            [instruction, {"mime_type": mime_type, "data": image_bytes}],
            request_options={"timeout": timeout},
        )
    except Exception as exc:  # pragma: no cover - surface Gemini errors
        raise RuntimeError(f"Gemini image caption failed: {exc}") from exc

    caption = getattr(response, "text", None)
    if caption:
        return caption.strip()

    # Some responses only populate candidates
    if hasattr(response, "candidates"):
        for candidate in response.candidates or []:
            if not candidate.content or not getattr(candidate.content, "parts", None):
                continue
            for part in candidate.content.parts:
                text = getattr(part, "text", None)
                if text:
                    return text.strip()

    return "No caption returned"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemini image caption tester")
    parser.add_argument("image", help="Path to the local image to describe")
    parser.add_argument("--prompt", help="Optional custom prompt", default=None)
    args = parser.parse_args()

    print(describe_image(args.image, prompt=args.prompt))