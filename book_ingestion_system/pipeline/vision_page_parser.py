"""
Step 2 — Vision LLM page parser.

Sends a page image to a vision model and returns raw extracted text/JSON.
Supports OpenAI GPT-4o, Google Gemini, Anthropic Claude, and Hugging Face local models.
"""

import base64
import json
import logging
from pathlib import Path

from config import (
    VISION_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    HF_MODEL,
)

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are an expert at extracting structured questions from textbook page images.

Analyze this page image and extract ALL questions visible on the page.

For each question return a JSON object with these exact keys:
- "question_number": the number shown beside the question (string)
- "question_text": the full question text, preserving any mathematical expressions
- "option_a": text of option (a)
- "option_b": text of option (b)
- "option_c": text of option (c)
- "option_d": text of option (d)

RULES:
- Preserve mathematical expressions exactly as written (fractions, exponents, roots, etc.)
- Ignore page headers, section titles, page numbers, and decorative text.
- If a question has fewer than 4 options, still include all option keys but set missing ones to "".
- Return ONLY a JSON array of question objects. No markdown fences, no explanation.
- If no questions are found on this page, return an empty array: []

Return valid JSON only."""

ANSWER_KEY_PROMPT = """You are an expert at extracting answer keys from textbook pages.

This page contains answer keys in formats like:
  331 (c)
  332 (b)
  333. (a)

Extract ALL answer mappings from this page.

Return a JSON object where:
- Keys are question numbers (strings)
- Values are the correct option letter in lowercase (a, b, c, or d)

Example: {"331": "c", "332": "b", "333": "a"}

RULES:
- Ignore any text that is not an answer mapping.
- Return ONLY a valid JSON object. No markdown fences, no explanation.
- If no answers are found, return an empty object: {}

Return valid JSON only."""


def _encode_image(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─── OpenAI ───────────────────────────────────────────────────────────────────

def _call_openai(image_path: Path, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    b64 = _encode_image(image_path)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    return response.choices[0].message.content


# ─── Gemini ───────────────────────────────────────────────────────────────────

def _call_gemini(image_path: Path, prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    with open(image_path, "rb") as f:
        image_data = f.read()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_data, mime_type="image/png"),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
        ),
    )
    return response.text


# ─── Anthropic ────────────────────────────────────────────────────────────────

def _call_anthropic(image_path: Path, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    b64 = _encode_image(image_path)

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.content[0].text


# ─── Dispatcher ───────────────────────────────────────────────────────────────

# ─── Hugging Face (local) ─────────────────────────────────────────────────────

_hf_pipe = None  # lazy-loaded singleton


def _get_hf_pipe():
    """Lazy-load the HuggingFace pipeline once and cache it."""
    global _hf_pipe
    if _hf_pipe is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading HuggingFace model %s (first call — this may take a while) …", HF_MODEL)
        _hf_pipe = hf_pipeline("image-text-to-text", model=HF_MODEL)
        logger.info("HuggingFace model loaded.")
    return _hf_pipe


def _call_huggingface(image_path: Path, prompt: str) -> str:
    pipe = _get_hf_pipe()
    img_uri = image_path.resolve().as_uri()  # file:///... URI

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": img_uri},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    result = pipe(text=messages, max_new_tokens=4096)
    # The pipeline returns a list of dicts; extract generated text
    if isinstance(result, list) and len(result) > 0:
        out = result[0]
        if isinstance(out, dict) and "generated_text" in out:
            gen = out["generated_text"]
            # generated_text can be a list of message dicts
            if isinstance(gen, list):
                # Find the last assistant message
                for msg in reversed(gen):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return msg.get("content", "")
                # fallback: join all text
                return str(gen)
            return str(gen)
    return str(result)


_PROVIDERS = {
    "openai": _call_openai,
    "gemini": _call_gemini,
    "anthropic": _call_anthropic,
    "huggingface": _call_huggingface,
}


def call_vision_model(image_path: Path, prompt: str, provider: str = VISION_PROVIDER) -> str:
    """Send an image + prompt to the configured vision LLM and return the raw text response."""
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"Unknown vision provider: {provider!r}. Choose from {list(_PROVIDERS)}")
    return fn(image_path, prompt)


def parse_page_questions(image_path: Path) -> list[dict]:
    """Extract questions from a single page image. Returns list of raw question dicts."""
    raw = call_vision_model(image_path, EXTRACTION_PROMPT)
    # Strip potential markdown code fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("JSON decode failed for %s — raw response:\n%s", image_path.name, raw[:500])
        return []
    if not isinstance(data, list):
        logger.error("Expected list from vision model, got %s", type(data).__name__)
        return []
    return data


def parse_answer_key_page(image_path: Path) -> dict[str, str]:
    """Extract answer key mappings from a single page image."""
    raw = call_vision_model(image_path, ANSWER_KEY_PROMPT)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Answer key JSON decode failed for %s — raw:\n%s", image_path.name, raw[:500])
        return {}
    if not isinstance(data, dict):
        logger.error("Expected dict from answer key parse, got %s", type(data).__name__)
        return {}
    return data
