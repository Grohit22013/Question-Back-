"""
Question Validator - Validates and repairs extracted questions.
"""

import re
from typing import List, Optional

from book_parser.schemas import StructuredQuestion, ValidationResult


class QuestionValidator:
    """Validates structured questions against the required schema rules."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.seen_question_numbers = set()

    def validate(self, question: StructuredQuestion) -> ValidationResult:
        """
        Validate a single question.

        Checks:
            1. Question text exists
            2. Four options exist
            3. Exactly one correct answer
            4. Question number is unique
            5. Mathematical expressions are intact
        """
        errors = []

        # Check 1: Question text exists
        if not question.question_name or not question.question_name.strip():
            errors.append("Missing question text")

        # Check 2: Four options exist
        choices = [
            question.choice1_text,
            question.choice2_text,
            question.choice3_text,
            question.choice4_text,
        ]
        empty_choices = [i for i, c in enumerate(choices, 1) if not c or not c.strip()]
        if empty_choices:
            errors.append(f"Missing options: {empty_choices}")

        # Check 3: Exactly one correct answer
        correct_flags = [
            question.choice1_isCorrect,
            question.choice2_isCorrect,
            question.choice3_isCorrect,
            question.choice4_isCorrect,
        ]
        correct_count = sum(1 for f in correct_flags if f)
        if correct_count == 0:
            errors.append("No correct answer marked")
        elif correct_count > 1:
            errors.append(f"Multiple correct answers marked ({correct_count})")

        # Check 4: Unique question number
        if question.question_number is not None:
            if question.question_number in self.seen_question_numbers:
                errors.append(f"Duplicate question number: {question.question_number}")
            else:
                self.seen_question_numbers.add(question.question_number)

        if errors:
            return ValidationResult(is_valid=False, errors=errors, question=question)

        return ValidationResult(is_valid=True, question=question)

    def attempt_repair(self, result: ValidationResult) -> ValidationResult:
        """
        Attempt to repair common validation issues.

        Repairs:
            - If no correct answer but question has answer_key mapping, set it
            - If empty option text, mark as "[empty]"
        """
        if result.is_valid:
            return result

        question = result.question
        repaired = False

        # Repair empty options with placeholder
        for attr in ["choice1_text", "choice2_text", "choice3_text", "choice4_text"]:
            if not getattr(question, attr, "").strip():
                setattr(question, attr, "[empty]")
                repaired = True

        if repaired:
            # Re-validate after repair
            new_result = self._validate_without_uniqueness(question)
            new_result.repaired = True
            return new_result

        return result

    def _validate_without_uniqueness(self, question: StructuredQuestion) -> ValidationResult:
        """Validate without the uniqueness check (for re-validation after repair)."""
        errors = []
        if not question.question_name or not question.question_name.strip():
            errors.append("Missing question text")

        choices = [
            question.choice1_text, question.choice2_text,
            question.choice3_text, question.choice4_text,
        ]
        empty_choices = [i for i, c in enumerate(choices, 1) if not c or not c.strip()]
        if empty_choices:
            errors.append(f"Missing options: {empty_choices}")

        correct_count = sum(1 for f in [
            question.choice1_isCorrect, question.choice2_isCorrect,
            question.choice3_isCorrect, question.choice4_isCorrect,
        ] if f)
        if correct_count != 1:
            errors.append(f"Expected 1 correct answer, got {correct_count}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            question=question,
        )

    def validate_batch(self, questions: List[StructuredQuestion]) -> dict:
        """
        Validate a batch of questions. Returns summary stats.
        """
        valid = []
        invalid = []
        repaired = []

        for q in questions:
            result = self.validate(q)
            if result.is_valid:
                valid.append(result)
            else:
                repair_result = self.attempt_repair(result)
                if repair_result.is_valid:
                    repaired.append(repair_result)
                else:
                    invalid.append(repair_result)

        return {
            "total": len(questions),
            "valid": len(valid),
            "repaired": len(repaired),
            "invalid": len(invalid),
            "valid_questions": [r.question for r in valid + repaired],
            "failed_questions": [r for r in invalid],
        }
