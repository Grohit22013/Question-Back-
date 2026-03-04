"""
Data schemas for the question bank system.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class RawExtraction:
    """Raw text extracted from a page image by the Vision Agent."""
    page_number: int
    raw_text: str
    image_path: str


@dataclass
class ParsedQuestion:
    """A single parsed question before answer mapping."""
    question_number: Optional[int]
    question_text: str
    choice1: str
    choice2: str
    choice3: str
    choice4: str
    page_number: int


@dataclass
class AnswerMapping:
    """Mapping of question number to correct answer."""
    question_number: int
    correct_option: str  # "a", "b", "c", or "d"


@dataclass
class StructuredQuestion:
    """Final structured question matching the output schema."""
    question_name: str = ""
    question_type_id: int = 1
    tags: str = ""
    weightage: str = ""
    difficulty: str = "medium"
    is_active: bool = True
    choice1_text: str = ""
    choice1_isCorrect: bool = False
    choice2_text: str = ""
    choice2_isCorrect: bool = False
    choice3_text: str = ""
    choice3_isCorrect: bool = False
    choice4_text: str = ""
    choice4_isCorrect: bool = False
    question_number: Optional[int] = None
    page_number: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "question_name": self.question_name,
            "question_type_id": self.question_type_id,
            "tags": self.tags,
            "weightage": self.weightage,
            "difficulty": self.difficulty,
            "is_active": self.is_active,
            "choice1_text": self.choice1_text,
            "choice1_isCorrect": self.choice1_isCorrect,
            "choice2_text": self.choice2_text,
            "choice2_isCorrect": self.choice2_isCorrect,
            "choice3_text": self.choice3_text,
            "choice3_isCorrect": self.choice3_isCorrect,
            "choice4_text": self.choice4_text,
            "choice4_isCorrect": self.choice4_isCorrect,
        }


@dataclass
class ValidationResult:
    """Result of validating a question."""
    is_valid: bool
    errors: list = field(default_factory=list)
    question: Optional[StructuredQuestion] = None
    repaired: bool = False
