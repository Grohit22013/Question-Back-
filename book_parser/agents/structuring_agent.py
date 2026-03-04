"""
Structuring Agent - LangChain agent that converts raw extracted text
into structured JSON questions.
"""

from typing import List, Dict

from book_parser.config import ModelConfig, TopicMapping
from book_parser.pipeline.question_parser import QuestionParser
from book_parser.schemas import RawExtraction, StructuredQuestion, ParsedQuestion


class StructuringAgent:
    """
    LangChain-powered agent that structures raw extraction text into
    clean, structured question objects.
    """

    def __init__(self, config: ModelConfig, topic_mapping: TopicMapping):
        self.config = config
        self.parser = QuestionParser(config)
        self.topic_mapping = topic_mapping

    def structure_extraction(self, extraction: RawExtraction) -> List[StructuredQuestion]:
        """
        Convert a single RawExtraction into structured questions.

        First tries LLM-based parsing, falls back to regex if needed.
        """
        # Try LLM-based parsing first
        parsed = self.parser.parse(extraction.raw_text, extraction.page_number)

        # Fallback to regex if LLM parsing yields nothing
        if not parsed:
            parsed = self.parser.parse_with_regex(
                extraction.raw_text, extraction.page_number
            )

        # Convert to StructuredQuestion objects
        topic = self.topic_mapping.get_topic(extraction.page_number)
        structured = []
        for p in parsed:
            sq = self._to_structured(p, topic)
            structured.append(sq)

        return structured

    def structure_batch(self, extractions: List[RawExtraction]) -> List[StructuredQuestion]:
        """
        Structure a batch of raw extractions into questions.
        """
        all_questions = []
        total = len(extractions)

        for i, extraction in enumerate(extractions):
            print(f"[StructuringAgent] Structuring page {extraction.page_number} ({i+1}/{total})...")
            questions = self.structure_extraction(extraction)
            all_questions.extend(questions)
            print(f"  -> Extracted {len(questions)} questions from page {extraction.page_number}")

        print(f"[StructuringAgent] Total structured questions: {len(all_questions)}")
        return all_questions

    def _to_structured(self, parsed: ParsedQuestion, topic: str) -> StructuredQuestion:
        """Convert a ParsedQuestion to a StructuredQuestion."""
        return StructuredQuestion(
            question_name=parsed.question_text,
            question_type_id=1,
            tags=topic,
            weightage="1",
            is_active=True,
            choice1_text=parsed.choice1,
            choice2_text=parsed.choice2,
            choice3_text=parsed.choice3,
            choice4_text=parsed.choice4,
            question_number=parsed.question_number,
            page_number=parsed.page_number,
        )
