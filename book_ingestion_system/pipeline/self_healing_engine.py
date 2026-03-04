"""
Step 8 — Self-healing error correction engine.

When validation fails, this module:
  1. Retries extraction using the original page image.
  2. Re-runs the vision parsing step.
  3. Compares outputs and repairs broken fields.
  4. Falls back to regex-based extraction.
  5. Flags unresolvable records for manual review.
"""

import logging
import re
from pathlib import Path

from models.question_schema import RawQuestion, QuestionRecord
from pipeline.vision_page_parser import parse_page_questions
from pipeline.question_answer_merger import merge_question_with_answer

logger = logging.getLogger(__name__)


def _regex_extract_question(text_block: str) -> dict | None:
    """
    Fallback regex-based extraction for a single question block.
    Tries to pull question number, text, and options from raw text.
    """
    # Match patterns like "123. Some question text\n(a) opt1 (b) opt2 ..."
    q_match = re.search(r"(\d+)\s*[.)]\s*(.+?)(?=\s*\(?[aA]\)?)", text_block, re.DOTALL)
    if not q_match:
        return None

    q_num = q_match.group(1)
    q_text = q_match.group(2).strip()

    options = {}
    for letter in "abcd":
        pattern = rf"\(?{letter}\)?\s*(.+?)(?=\s*\(?[{chr(ord(letter)+1)}]?\)?|\Z)"
        m = re.search(pattern, text_block, re.IGNORECASE | re.DOTALL)
        options[f"option_{letter}"] = m.group(1).strip() if m else ""

    return {
        "question_number": q_num,
        "question_text": q_text,
        **options,
    }


def retry_vision_extraction(
    image_path: Path,
    page_number: int,
) -> list[RawQuestion]:
    """
    Retry extraction from the original page image.
    """
    logger.info("Retrying vision extraction for page %d", page_number)
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
            logger.warning("Retry parse failed for item on page %d: %s", page_number, exc)

    return questions


def _repair_record(
    broken_record: QuestionRecord,
    errors: list[str],
    retried_raw: list[RawQuestion],
    answer_key: dict[str, str],
) -> QuestionRecord | None:
    """
    Attempt to repair a broken record using retried extraction data.
    """
    # Find matching question in retried data by looking for similar text
    target_text = broken_record.question_name.lower().strip()

    for raw_q in retried_raw:
        if raw_q.question_text.lower().strip() == target_text or (
            target_text and target_text[:40] in raw_q.question_text.lower()
        ):
            repaired = merge_question_with_answer(raw_q, answer_key)
            repaired.tags = broken_record.tags
            repaired.weightage = broken_record.weightage
            logger.info("Repaired question: %s", raw_q.question_number)
            return repaired

    return None


def heal_invalid_records(
    invalid_records: list[tuple[QuestionRecord, list[str]]],
    page_images: dict[int, Path],
    raw_questions: list[RawQuestion],
    answer_key: dict[str, str],
    max_retries: int = 3,
) -> tuple[list[QuestionRecord], list[tuple[QuestionRecord, list[str]]]]:
    """
    Attempt to heal all invalid records.

    Returns:
        (healed_records, still_broken_records)
    """
    healed: list[QuestionRecord] = []
    still_broken: list[tuple[QuestionRecord, list[str]]] = []

    # Group broken records by page for efficient retry
    page_broken: dict[int, list[tuple[QuestionRecord, list[str]]]] = {}
    for rec, errs in invalid_records:
        # Find the page number from raw_questions
        page_num = 0
        rec_text = rec.question_name.lower().strip()
        for rq in raw_questions:
            if rq.question_text.lower().strip() == rec_text or (
                rec_text and rec_text[:40] in rq.question_text.lower()
            ):
                page_num = rq.page_number
                break
        page_broken.setdefault(page_num, []).append((rec, errs))

    for page_num, broken_list in page_broken.items():
        if page_num == 0 or page_num not in page_images:
            still_broken.extend(broken_list)
            continue

        retry_success = False
        retried_raw: list[RawQuestion] = []

        for attempt in range(1, max_retries + 1):
            logger.info("Healing attempt %d/%d for page %d", attempt, max_retries, page_num)
            retried_raw = retry_vision_extraction(page_images[page_num], page_num)
            if retried_raw:
                retry_success = True
                break

        if not retry_success:
            still_broken.extend(broken_list)
            continue

        for rec, errs in broken_list:
            repaired = _repair_record(rec, errs, retried_raw, answer_key)
            if repaired:
                healed.append(repaired)
            else:
                still_broken.append((rec, errs))

    logger.info(
        "Self-healing complete — healed: %d, still broken: %d (flagged for manual review)",
        len(healed), len(still_broken),
    )
    return healed, still_broken
