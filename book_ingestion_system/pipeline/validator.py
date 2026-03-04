"""
Step 7 — Data validator.

Validates every QuestionRecord against the rules defined in the masterprompt:
  1. Question text must exist.
  2. Exactly four options must exist.
  3. Only one option must be marked correct.
  4. Question numbers must be unique.
  5. Mathematical expressions must not be corrupted.
"""

import hashlib
import logging
import re

from models.question_schema import QuestionRecord

logger = logging.getLogger(__name__)


class ValidationResult:
    def __init__(self):
        self.valid: list[QuestionRecord] = []
        self.invalid: list[tuple[QuestionRecord, list[str]]] = []

    @property
    def total(self) -> int:
        return len(self.valid) + len(self.invalid)


def _has_corrupted_math(text: str) -> bool:
    """Detect likely corruption in mathematical expressions."""
    # Unbalanced parentheses / brackets
    if text.count("(") != text.count(")"):
        return True
    if text.count("[") != text.count("]"):
        return True
    # Lone operators at start/end (likely truncated)
    stripped = text.strip()
    if stripped and stripped[-1] in "+-×÷*/^=<>" and len(stripped) > 1:
        return True
    # Unicode replacement character
    if "\ufffd" in text:
        return True
    return False


def _question_hash(text: str) -> str:
    """Compute a normalized hash for duplicate detection."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()


def validate_records(records: list[QuestionRecord]) -> ValidationResult:
    """
    Run all validation rules. Returns a ValidationResult with valid / invalid splits.
    """
    result = ValidationResult()
    seen_hashes: set[str] = set()

    for rec in records:
        errors: list[str] = []

        # Rule 1: Question text must exist
        if not rec.question_name or not rec.question_name.strip():
            errors.append("missing_question_text")

        # Rule 2: Exactly four options must exist (non-empty)
        choices = [rec.choice1_text, rec.choice2_text, rec.choice3_text, rec.choice4_text]
        non_empty = [c for c in choices if c and c.strip()]
        if len(non_empty) < 4:
            errors.append(f"incomplete_options({len(non_empty)}/4)")

        # Rule 3: Exactly one correct answer
        correct_count = sum([
            rec.choice1_isCorrect,
            rec.choice2_isCorrect,
            rec.choice3_isCorrect,
            rec.choice4_isCorrect,
        ])
        if correct_count != 1:
            errors.append(f"incorrect_answer_count({correct_count})")

        # Rule 5: Math expression integrity
        all_text = " ".join([rec.question_name] + choices)
        if _has_corrupted_math(all_text):
            errors.append("corrupted_math_expression")

        # Rule 4: Duplicate detection
        q_hash = _question_hash(rec.question_name)
        if q_hash in seen_hashes:
            errors.append("duplicate_question")
        else:
            seen_hashes.add(q_hash)

        if errors:
            result.invalid.append((rec, errors))
        else:
            result.valid.append(rec)

    logger.info(
        "Validation complete — %d valid, %d invalid out of %d total",
        len(result.valid), len(result.invalid), result.total,
    )
    if result.invalid:
        error_summary: dict[str, int] = {}
        for _, errs in result.invalid:
            for e in errs:
                key = e.split("(")[0]
                error_summary[key] = error_summary.get(key, 0) + 1
        logger.info("Validation error breakdown: %s", error_summary)

    return result
