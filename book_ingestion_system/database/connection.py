"""
Database connection manager using psycopg2 with connection pooling.
"""

import logging

import psycopg2
from psycopg2 import pool

from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

logger = logging.getLogger(__name__)

_connection_pool: pool.ThreadedConnectionPool | None = None


def init_pool(min_conn: int = 2, max_conn: int = 10):
    """Initialize the connection pool."""
    global _connection_pool
    if _connection_pool is not None:
        return

    _connection_pool = pool.ThreadedConnectionPool(
        min_conn,
        max_conn,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    logger.info("Database connection pool initialized (host=%s, db=%s)", DB_HOST, DB_NAME)


def get_connection():
    """Get a connection from the pool."""
    if _connection_pool is None:
        init_pool()
    return _connection_pool.getconn()


def release_connection(conn):
    """Return a connection to the pool."""
    if _connection_pool is not None:
        _connection_pool.putconn(conn)


def close_pool():
    """Shut down the connection pool."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Database connection pool closed.")


def ensure_table():
    """Create the questions table if it doesn't exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id SERIAL PRIMARY KEY,
                    question_name TEXT NOT NULL,
                    question_type_id INTEGER DEFAULT 1,
                    tags VARCHAR(255) DEFAULT '',
                    weightage VARCHAR(50) DEFAULT '',
                    is_active BOOLEAN DEFAULT TRUE,
                    choice1_text TEXT DEFAULT '',
                    choice1_iscorrect BOOLEAN DEFAULT FALSE,
                    choice2_text TEXT DEFAULT '',
                    choice2_iscorrect BOOLEAN DEFAULT FALSE,
                    choice3_text TEXT DEFAULT '',
                    choice3_iscorrect BOOLEAN DEFAULT FALSE,
                    choice4_text TEXT DEFAULT '',
                    choice4_iscorrect BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_questions_tags ON questions (tags);
            """)
        conn.commit()
        logger.info("Table 'questions' ensured.")
    finally:
        release_connection(conn)
