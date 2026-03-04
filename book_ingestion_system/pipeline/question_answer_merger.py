"""
Step 4 — Merge extracted questions with answer keys.

Maps the correct option letter to the corresponding choice field
and produces QuestionRecord objects ready for the database.
"""

import logging

from models.question_schema import RawQuestion, QuestionRecord

logger = logging.getLogger(__name__)

_OPTION_MAP = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
}


def merge_question_with_answer(
    question: RawQuestion,
    answer_key: dict[str, str],
) -> QuestionRecord:
    """
    Build a QuestionRecord by merging a RawQuestion with its correct answer.
    """
    correct_letter = answer_key.get(question.question_number, "").lower()
    correct_index = _OPTION_MAP.get(correct_letter, 0)

    return QuestionRecord(
        question_name=question.question_text,
        question_type_id=1,
        tags="",
        weightage="",
        is_active=True,
        choice1_text=question.option_a,
        choice1_isCorrect=(correct_index == 1),
        choice2_text=question.option_b,
        choice2_isCorrect=(correct_index == 2),
        choice3_text=question.option_c,
        choice3_isCorrect=(correct_index == 3),
        choice4_text=question.option_d,
        choice4_isCorrect=(correct_index == 4),
    )


def merge_all(
    questions: list[RawQuestion],
    answer_key: dict[str, str],
) -> list[QuestionRecord]:
    """
    Merge every extracted question with its answer key entry.
    """
    records: list[QuestionRecord] = []

    for q in questions:
        record = merge_question_with_answer(q, answer_key)
        records.append(record)

    matched = sum(
        1 for r in records
        if any([
            r.choice1_isCorrect,
            r.choice2_isCorrect,
            r.choice3_isCorrect,
            r.choice4_isCorrect,
        ])
    )
    logger.info(
        "Merged %d questions — %d with matched answers, %d unmatched",
        len(records), matched, len(records) - matched,
    )
    return records
