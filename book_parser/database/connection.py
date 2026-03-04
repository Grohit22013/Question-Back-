"""
PostgreSQL database connection and management.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from book_parser.config import DatabaseConfig


# SQL to create the questions table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS questions (
    id SERIAL PRIMARY KEY,
    question_name TEXT NOT NULL,
    question_type_id INTEGER DEFAULT 1,
    tags VARCHAR(255),
    weightage VARCHAR(10),
    difficulty VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    choice1_text TEXT,
    choice1_isCorrect BOOLEAN DEFAULT FALSE,
    choice2_text TEXT,
    choice2_isCorrect BOOLEAN DEFAULT FALSE,
    choice3_text TEXT,
    choice3_isCorrect BOOLEAN DEFAULT FALSE,
    choice4_text TEXT,
    choice4_isCorrect BOOLEAN DEFAULT FALSE,
    question_number INTEGER,
    page_number INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_questions_tags ON questions(tags);
CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions(difficulty);
CREATE INDEX IF NOT EXISTS idx_questions_number ON questions(question_number);
"""


class DatabaseConnection:
    """PostgreSQL database connection manager."""

    def __init__(self, config: DatabaseConfig):
        self.config = config

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def get_cursor(self, dict_cursor: bool = False):
        """Context manager for database cursors."""
        cursor_factory = RealDictCursor if dict_cursor else None
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def initialize(self):
        """Create the questions table if it doesn't exist."""
        with self.get_cursor() as cursor:
            cursor.execute(CREATE_TABLE_SQL)
        print("Database initialized: questions table ready.")

    def get_question_count(self) -> int:
        """Return the total number of questions in the database."""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM questions")
            return cursor.fetchone()[0]
