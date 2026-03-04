"""
Bulk insert operations for the questions database.
"""

from typing import List

from book_parser.database.connection import DatabaseConnection
from book_parser.schemas import StructuredQuestion


INSERT_SQL = """
INSERT INTO questions (
    question_name, question_type_id, tags, weightage, difficulty, is_active,
    choice1_text, choice1_isCorrect,
    choice2_text, choice2_isCorrect,
    choice3_text, choice3_isCorrect,
    choice4_text, choice4_isCorrect,
    question_number, page_number
) VALUES (
    %(question_name)s, %(question_type_id)s, %(tags)s, %(weightage)s,
    %(difficulty)s, %(is_active)s,
    %(choice1_text)s, %(choice1_isCorrect)s,
    %(choice2_text)s, %(choice2_isCorrect)s,
    %(choice3_text)s, %(choice3_isCorrect)s,
    %(choice4_text)s, %(choice4_isCorrect)s,
    %(question_number)s, %(page_number)s
)
"""


def bulk_insert(db: DatabaseConnection, questions: List[StructuredQuestion], batch_size: int = 100) -> int:
    """
    Insert questions into the database in batches.

    Args:
        db: Database connection instance.
        questions: List of StructuredQuestion objects.
        batch_size: Number of questions per batch.

    Returns:
        Total number of questions inserted.
    """
    total_inserted = 0

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        params = [
            {
                "question_name": q.question_name,
                "question_type_id": q.question_type_id,
                "tags": q.tags,
                "weightage": q.weightage,
                "difficulty": q.difficulty,
                "is_active": q.is_active,
                "choice1_text": q.choice1_text,
                "choice1_isCorrect": q.choice1_isCorrect,
                "choice2_text": q.choice2_text,
                "choice2_isCorrect": q.choice2_isCorrect,
                "choice3_text": q.choice3_text,
                "choice3_isCorrect": q.choice3_isCorrect,
                "choice4_text": q.choice4_text,
                "choice4_isCorrect": q.choice4_isCorrect,
                "question_number": q.question_number,
                "page_number": q.page_number,
            }
            for q in batch
        ]

        with db.get_cursor() as cursor:
            cursor.executemany(INSERT_SQL, params)

        total_inserted += len(batch)
        print(f"  Inserted batch: {total_inserted}/{len(questions)}")

    return total_inserted


def export_to_csv(questions: List[StructuredQuestion], output_path: str):
    """Export questions to a CSV file as a fallback."""
    import pandas as pd

    rows = []
    for q in questions:
        rows.append(q.to_dict())

    df = pd.DataFrame(rows)
    column_order = [
        "question_name", "question_type_id", "tags", "weightage", "is_active",
        "choice1_text", "choice1_isCorrect",
        "choice2_text", "choice2_isCorrect",
        "choice3_text", "choice3_isCorrect",
        "choice4_text", "choice4_isCorrect",
    ]
    df = df[column_order]
    df.to_csv(output_path, index=False)
    print(f"Exported {len(questions)} questions to {output_path}")
