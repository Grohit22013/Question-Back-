"""
Bulk insertion of QuestionRecords into PostgreSQL.
"""

import logging

from psycopg2.extras import execute_values

from models.question_schema import QuestionRecord
from database.connection import get_connection, release_connection

logger = logging.getLogger(__name__)

INSERT_SQL = """
    INSERT INTO questions (
        question_name, question_type_id, tags, weightage, is_active,
        choice1_text, choice1_iscorrect,
        choice2_text, choice2_iscorrect,
        choice3_text, choice3_iscorrect,
        choice4_text, choice4_iscorrect
    ) VALUES %s
"""


def _record_to_tuple(rec: QuestionRecord) -> tuple:
    return (
        rec.question_name,
        rec.question_type_id,
        rec.tags,
        rec.weightage,
        rec.is_active,
        rec.choice1_text,
        rec.choice1_isCorrect,
        rec.choice2_text,
        rec.choice2_isCorrect,
        rec.choice3_text,
        rec.choice3_isCorrect,
        rec.choice4_text,
        rec.choice4_isCorrect,
    )


def bulk_insert(records: list[QuestionRecord], batch_size: int = 1000) -> int:
    """
    Insert records in batches using execute_values for performance.
    Returns the total number of records inserted.
    """
    if not records:
        logger.info("No records to insert.")
        return 0

    conn = get_connection()
    total_inserted = 0

    try:
        with conn.cursor() as cur:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                values = [_record_to_tuple(r) for r in batch]
                execute_values(cur, INSERT_SQL, values)
                total_inserted += len(batch)
                logger.info(
                    "Inserted batch %d–%d (%d records)",
                    i + 1, i + len(batch), len(batch),
                )
        conn.commit()
        logger.info("Bulk insert complete — %d total records.", total_inserted)
    except Exception:
        conn.rollback()
        logger.exception("Bulk insert failed — transaction rolled back.")
        raise
    finally:
        release_connection(conn)

    return total_inserted
