"""Gemini 2.5 Flash image captioning helper."""

from __future__ import annotations

import logging
import mimetypes
import os
import time
import re
import json
from pathlib import Path
from typing import Optional, List, Dict

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

__all__ = ["describe_image", "summarize_video", "search_memory_nodes", "generate_title", "generate_short_answer"]


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


def summarize_video(video_path: str, prompt: Optional[str] = None, timeout: int = 300) -> str:
    """Return a natural-language summary of a video file using the Gemini API.

    Args:
        video_path: Path to a local video file.
        prompt: Optional custom instruction.
        timeout: Seconds to wait for Gemini response.

    Raises:
        FileNotFoundError: if the video path does not exist.
        RuntimeError: if configuration or Gemini API calls fail.

    Returns:
        Summary text string. Falls back to "No summary returned" if model returns empty content.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = _get_model()
    instruction = prompt or "Summarize this video. Describe the key objects, people, and actions."

    LOGGER.info(f"Uploading video for analysis: {path.name}")
    video_file = genai.upload_file(path=path)

    # Wait for the video to be processed
    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise RuntimeError(f"Video processing failed for {path.name}")

    LOGGER.info(f"Video uploaded successfully. Generating summary for {path.name}...")

    try:
        response = model.generate_content(
            [instruction, video_file],
            request_options={"timeout": timeout},
        )
    except Exception as exc:
        # Clean up the uploaded file on error
        genai.delete_file(video_file.name)
        LOGGER.error(f"Gemini video summary failed: {exc}", exc_info=True)
        raise RuntimeError(f"Gemini video summary failed: {exc}") from exc

    # Clean up the uploaded file
    genai.delete_file(video_file.name)
    LOGGER.info(f"Cleaned up uploaded file: {video_file.name}")

    summary = getattr(response, "text", None)
    if summary:
        return summary.strip()

    return "No summary returned"


def search_memory_nodes(query: str, memory_nodes: List[Dict], max_results: int = 5, timeout: int = 60) -> List[Dict]:
    """Search through MemoryNodes using Gemini to find the most relevant ones based on a query.
    
    Args:
        query: User's search query
        memory_nodes: List of MemoryNode dictionaries from the database
        max_results: Maximum number of results to return
        timeout: Seconds to wait for Gemini response
    
    Returns:
        List of the most relevant MemoryNode dictionaries, ordered by relevance
    """
    if not memory_nodes:
        return []
    
    model = _get_model()
    
    # Format MemoryNodes for Gemini
    nodes_text = "MemoryNodes:\n"
    for i, node in enumerate(memory_nodes):
        metadata_str = ""
        if node.get('metadata'):
            try:
                metadata = json.loads(node['metadata']) if isinstance(node['metadata'], str) else node['metadata']
                metadata_str = f"Metadata: {json.dumps(metadata, indent=2)}\n"
            except:
                metadata_str = f"Metadata: {node.get('metadata', '')}\n"
        
        nodes_text += f"""
Node {node['id']}:
  File Type: {node['file_type']}
  File Path: {node['file_path']}
  Timestamp: {node['timestamp']}
  {metadata_str}
"""
    
    prompt = f"""You are a search assistant. Given the following MemoryNodes and a user query, identify the most relevant MemoryNodes.

{nodes_text}

User Query: "{query}"

Analyze the query and the MemoryNodes. Pay special attention to:
- Summary content: Compare the query with the summary field in metadata. The summary describes what happened in the video/event.
- Transcript content: Compare the query with the transcript field in metadata. The transcript contains the actual spoken words from the audio.
- Semantic matching: Look for conceptual matches, not just exact keyword matches. For example, if the query mentions "cooking" and a summary says "preparing a meal", these should be considered relevant.
- Objects detected: Check if the query mentions objects that appear in the objects_detected field.
- File type relevance (video, audio, transcript)
- Timestamp relevance (if the query mentions time-related information)
- Title: Check if the query matches the title field in metadata

The summary and transcript are the most important fields for determining relevance. A MemoryNode is relevant if:
- Its summary describes events, people, or activities related to the query
- Its transcript contains words or topics related to the query
- The query's intent matches what the event captured

Return ONLY a JSON array of the node IDs (as integers) that are most relevant to the query, ordered by relevance (most relevant first). 
Return at most {max_results} node IDs.
Format: [id1, id2, id3, ...]

Example response: [5, 12, 3]"""

    try:
        response = model.generate_content(
            prompt,
            request_options={"timeout": timeout},
        )
        
        response_text = getattr(response, "text", None)
        if not response_text:
            # Try to extract from candidates
            if hasattr(response, "candidates"):
                for candidate in response.candidates or []:
                    if not candidate.content or not getattr(candidate.content, "parts", None):
                        continue
                    for part in candidate.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            response_text = text
                            break
        
        if response_text:
            # Try to parse JSON array from response
            # Remove markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            elif response_text.startswith("`"):
                response_text = response_text.strip("`")
            
            # Extract JSON array
            json_match = re.search(r'\[[\d\s,]+\]', response_text)
            if json_match:
                node_ids = json.loads(json_match.group())
            else:
                # Try to parse the whole response as JSON
                node_ids = json.loads(response_text)
            
            # Create a mapping of node IDs to nodes
            node_map = {node['id']: node for node in memory_nodes}
            
            # Return nodes in the order specified by Gemini
            results = []
            for node_id in node_ids:
                if node_id in node_map:
                    results.append(node_map[node_id])
            
            return results[:max_results]
        
    except Exception as exc:
        LOGGER.error(f"Gemini memory node search failed: {exc}", exc_info=True)
        # Fallback: return empty list or all nodes if search fails
        return []
    
    return []


def generate_title(summary: str, timeout: int = 30) -> str:
    """Generate a very short title (max 50 characters) from a summary using Gemini.
    
    Args:
        summary: The full summary text to create a title from
        timeout: Seconds to wait for Gemini response
    
    Returns:
        A short title string (max 50 characters), or a fallback if generation fails
    """
    if not summary or len(summary.strip()) == 0:
        return "Recording"
    
    model = _get_model()
    
    prompt = f"""Generate a very short, concise title (maximum 50 characters) that summarizes the following text.

Text: "{summary}"

Return ONLY the title, nothing else. The title should be:
- Very concise (50 characters maximum)
- Descriptive of the main event or activity
- Natural and readable
- No quotes or formatting

Example: "Person cooking dinner in kitchen"

Title:"""

    try:
        response = model.generate_content(
            prompt,
            request_options={"timeout": timeout},
        )
        
        title = getattr(response, "text", None)
        if title:
            title = title.strip()
            # Remove quotes if present
            title = title.strip('"').strip("'").strip()
            # Truncate to 50 characters if needed
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        
        # Try to extract from candidates
        if hasattr(response, "candidates"):
            for candidate in response.candidates or []:
                if not candidate.content or not getattr(candidate.content, "parts", None):
                    continue
                for part in candidate.content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        title = text.strip().strip('"').strip("'").strip()
                        if len(title) > 50:
                            title = title[:47] + "..."
                        return title
        
    except Exception as exc:
        LOGGER.warning(f"Gemini title generation failed: {exc}. Using fallback.")
    
    # Fallback: create a simple title from the summary
    fallback = summary.strip()
    if len(fallback) > 50:
        # Find the last space before 50 characters
        last_space = fallback[:47].rfind(' ')
        if last_space > 20:
            fallback = fallback[:last_space] + "..."
        else:
            fallback = fallback[:47] + "..."
    
    return fallback


def generate_short_answer(query: str, summary: str, timeout: int = 30) -> str:
    """Generate a short answer to a user query based on an event summary using Gemini.
    
    Args:
        query: User's question/query
        summary: The event summary to base the answer on
        timeout: Seconds to wait for Gemini response
    
    Returns:
        A short answer string
    """
    if not query or not summary:
        return "I don't have enough information to answer that question."
    
    model = _get_model()
    
    prompt = f"""Based on the following event summary, provide a concise and direct answer to the user's question.

Event Summary:
{summary}

User Question: "{query}"

Provide a brief, natural answer (2-3 sentences maximum) that directly addresses the question based on the event summary. Be conversational and helpful."""

    try:
        response = model.generate_content(
            prompt,
            request_options={"timeout": timeout},
        )
        
        answer = getattr(response, "text", None)
        if answer:
            return answer.strip()
        
        # Try to extract from candidates
        if hasattr(response, "candidates"):
            for candidate in response.candidates or []:
                if not candidate.content or not getattr(candidate.content, "parts", None):
                    continue
                for part in candidate.content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        return text.strip()
        
    except Exception as exc:
        LOGGER.error(f"Gemini answer generation failed: {exc}", exc_info=True)
    
    return "I'm sorry, I couldn't generate an answer to that question."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemini image caption tester")
    parser.add_argument("image", help="Path to the local image to describe")
    parser.add_argument("--prompt", help="Optional custom prompt", default=None)
    args = parser.parse_args()

    print(describe_image(args.image, prompt=args.prompt))