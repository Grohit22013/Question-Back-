"""
Pydantic models for the question bank schema.
"""

from pydantic import BaseModel, field_validator


class RawQuestion(BaseModel):
    """Intermediate representation straight from vision extraction."""
    question_number: str
    question_text: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    page_number: int = 0


class QuestionRecord(BaseModel):
    """Final database-ready record matching the target schema."""
    question_name: str
    question_type_id: int = 1
    tags: str = ""
    weightage: str = ""  # easy / medium / hard
    is_active: bool = True
    choice1_text: str
    choice1_isCorrect: bool = False
    choice2_text: str
    choice2_isCorrect: bool = False
    choice3_text: str
    choice3_isCorrect: bool = False
    choice4_text: str
    choice4_isCorrect: bool = False

    @field_validator("weightage")
    @classmethod
    def validate_weightage(cls, v: str) -> str:
        allowed = {"easy", "medium", "hard", ""}
        if v.lower() not in allowed:
            raise ValueError(f"weightage must be one of {allowed}")
        return v.lower()
