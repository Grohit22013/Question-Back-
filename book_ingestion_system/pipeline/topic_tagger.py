"""
Step 5 — Topic tagger.

Assigns topic tags to questions based on page-range mappings from the book index.
"""

import logging

from models.question_schema import RawQuestion, QuestionRecord
from config import TOPIC_MAP

logger = logging.getLogger(__name__)


def get_topic_for_page(page_number: int, topic_map: list[tuple[int, int, str]] = TOPIC_MAP) -> str:
    """Return the topic tag for a given page number."""
    for start, end, topic in topic_map:
        if start <= page_number <= end:
            return topic
    return "general"


def assign_topics(
    records: list[QuestionRecord],
    raw_questions: list[RawQuestion],
) -> list[QuestionRecord]:
    """
    Assign topic tags to each QuestionRecord based on the page number
    stored in the corresponding RawQuestion.
    """
    if len(records) != len(raw_questions):
        logger.warning(
            "Record count (%d) != raw question count (%d). "
            "Tagging only the first min(len) entries.",
            len(records), len(raw_questions),
        )

    for i in range(min(len(records), len(raw_questions))):
        page_num = raw_questions[i].page_number
        topic = get_topic_for_page(page_num)
        records[i].tags = topic

    tagged_counts: dict[str, int] = {}
    for r in records:
        tagged_counts[r.tags] = tagged_counts.get(r.tags, 0) + 1

    logger.info("Topic tagging complete — distribution: %s", tagged_counts)
    return records
