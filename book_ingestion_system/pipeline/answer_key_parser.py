"""
Step 3 — Answer key parser.

Extracts answer key mappings from answer-key pages and also supports
regex-based fallback for text-extracted answer keys.
"""

import logging
import re
from pathlib import Path

from pipeline.vision_page_parser import parse_answer_key_page

logger = logging.getLogger(__name__)

# Pattern: "331 (c)" or "331. (c)" or "331.(c)" or "331 c"
_ANSWER_PATTERN = re.compile(
    r"(\d+)\s*\.?\s*\(?([a-dA-D])\)?",
)


def parse_answer_keys_from_images(image_paths: list[Path]) -> dict[str, str]:
    """
    Send answer-key page images to the vision model and aggregate all mappings.
    """
    combined: dict[str, str] = {}

    for img_path in image_paths:
        page_answers = parse_answer_key_page(img_path)
        for qnum, ans in page_answers.items():
            combined[str(qnum).strip()] = ans.strip().lower()

    logger.info("Answer keys extracted via vision: %d entries", len(combined))
    return combined


def parse_answer_keys_from_text(text: str) -> dict[str, str]:
    """
    Fallback regex parser for answer key text.

    Accepts raw text containing lines like:
        331 (c)
        332 (b)
    """
    answers: dict[str, str] = {}
    for match in _ANSWER_PATTERN.finditer(text):
        qnum = match.group(1)
        ans = match.group(2).lower()
        answers[qnum] = ans

    logger.info("Answer keys extracted via regex: %d entries", len(answers))
    return answers


def load_answer_keys(
    image_paths: list[Path] | None = None,
    text: str | None = None,
) -> dict[str, str]:
    """
    Load answer keys from either image paths (vision) or raw text (regex fallback).
    """
    answers: dict[str, str] = {}

    if image_paths:
        answers.update(parse_answer_keys_from_images(image_paths))

    if text:
        regex_answers = parse_answer_keys_from_text(text)
        # Only add keys not already found via vision
        for k, v in regex_answers.items():
            if k not in answers:
                answers[k] = v

    logger.info("Total answer keys loaded: %d", len(answers))
    return answers
