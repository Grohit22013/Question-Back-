"""
Step 2b — Question extractor.

Converts raw vision output dicts into validated RawQuestion models
and handles per-page extraction orchestration.
"""

import logging
from pathlib import Path

from models.question_schema import RawQuestion
from pipeline.vision_page_parser import parse_page_questions

logger = logging.getLogger(__name__)


def extract_questions_from_page(image_path: Path, page_number: int) -> list[RawQuestion]:
    """
    Run vision extraction on a single page image and return a list of RawQuestion objects.
    """
    raw_dicts = parse_page_questions(image_path)
    questions: list[RawQuestion] = []

    for item in raw_dicts:
        try:
            q = RawQuestion(
                question_number=str(item.get("question_number", "")).strip(),
                question_text=str(item.get("question_text", "")).strip(),
                option_a=str(item.get("option_a", "")).strip(),
                option_b=str(item.get("option_b", "")).strip(),
                option_c=str(item.get("option_c", "")).strip(),
                option_d=str(item.get("option_d", "")).strip(),
                page_number=page_number,
            )
            questions.append(q)
        except Exception as exc:
            logger.warning(
                "Failed to parse question on page %d: %s — data: %s",
                page_number, exc, item,
            )

    logger.info("Page %d: extracted %d questions", page_number, len(questions))
    return questions


def extract_all_questions(
    image_paths: list[Path],
    start_page: int = 1,
) -> list[RawQuestion]:
    """
    Extract questions from all page images sequentially.

    `start_page` is the 1-based page number of the first image in the list.
    """
    all_questions: list[RawQuestion] = []

    for idx, img_path in enumerate(image_paths):
        page_num = start_page + idx
        page_questions = extract_questions_from_page(img_path, page_num)
        all_questions.extend(page_questions)

    logger.info("Total questions extracted: %d", len(all_questions))
    return all_questions
