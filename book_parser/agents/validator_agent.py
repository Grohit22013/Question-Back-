"""
Validator Agent - LangChain agent that validates and repairs
extracted questions before database insertion.
"""

from typing import List

from book_parser.pipeline.validator import QuestionValidator
from book_parser.schemas import StructuredQuestion, ValidationResult


class ValidatorAgent:
    """
    Agent that validates all extracted questions against the required schema.
    Attempts repairs for common issues and flags unrepairable questions.
    """

    def __init__(self, max_retries: int = 2):
        self.validator = QuestionValidator(max_retries=max_retries)

    def validate_all(self, questions: List[StructuredQuestion]) -> dict:
        """
        Validate all questions and return results summary.

        Returns:
            Dict with keys: total, valid, repaired, invalid,
            valid_questions, failed_questions
        """
        print(f"[ValidatorAgent] Validating {len(questions)} questions...")
        results = self.validator.validate_batch(questions)

        print(f"[ValidatorAgent] Results:")
        print(f"  Total:    {results['total']}")
        print(f"  Valid:    {results['valid']}")
        print(f"  Repaired: {results['repaired']}")
        print(f"  Invalid:  {results['invalid']}")

        if results['failed_questions']:
            print(f"\n[ValidatorAgent] Failed questions:")
            for r in results['failed_questions'][:5]:  # Show first 5
                q_num = r.question.question_number if r.question else "?"
                print(f"  Q{q_num}: {r.errors}")

        return results
