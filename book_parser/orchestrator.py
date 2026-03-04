"""
LangChain Multi-Agent Orchestrator Pipeline.

Orchestrates the full book-to-question-bank pipeline:
  PDF → Images → Vision Agent → Structuring Agent → Answer Agent →
  Validator Agent → Difficulty Classification → Database Insert
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Dict

from book_parser.config import AppConfig
from book_parser.schemas import StructuredQuestion

from book_parser.pipeline.pdf_to_images import pdf_to_images, get_page_number_from_path
from book_parser.pipeline.difficulty_classifier import DifficultyClassifier

from book_parser.agents.vision_agent import VisionAgent
from book_parser.agents.structuring_agent import StructuringAgent
from book_parser.agents.answer_agent import AnswerAgent
from book_parser.agents.validator_agent import ValidatorAgent

from book_parser.database.connection import DatabaseConnection
from book_parser.database.bulk_insert import bulk_insert, export_to_csv


class BookIngestionPipeline:
    """
    Main orchestrator that runs the full multi-agent pipeline.

    Pipeline stages:
    1. PDF → Page images
    2. Vision Agent: Image → Raw text
    3. Structuring Agent: Raw text → Structured questions
    4. Answer Agent: Answer key → Merge answers
    5. Validator Agent: Validate & repair
    6. Difficulty classification & topic tagging
    7. Database insertion / CSV export
    """

    def __init__(self, config: AppConfig):
        self.config = config
        config.ensure_dirs()

        # Initialize agents
        self.vision_agent = VisionAgent(config.model)
        self.structuring_agent = StructuringAgent(config.model, config.topics)
        self.answer_agent = AnswerAgent(config.model)
        self.validator_agent = ValidatorAgent()
        self.difficulty_classifier = DifficultyClassifier(config.model)

    def run(
        self,
        pdf_path: str,
        answer_key_pages: Optional[List[int]] = None,
        answer_key_text: Optional[str] = None,
        question_pages: Optional[tuple] = None,
        skip_db: bool = False,
        output_csv: str = "output/questions.csv",
        output_json: str = "output/questions.json",
    ) -> List[StructuredQuestion]:
        """
        Run the full ingestion pipeline.

        Args:
            pdf_path: Path to the input PDF book.
            answer_key_pages: List of page numbers containing the answer key.
            answer_key_text: Pre-extracted answer key text (alternative to pages).
            question_pages: Tuple (start, end) to limit which pages to process.
            skip_db: If True, skip database insertion.
            output_csv: Path for CSV export.
            output_json: Path for JSON export.

        Returns:
            List of valid StructuredQuestion objects.
        """
        start_time = time.time()

        # ─── Stage 1: PDF → Images ───────────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 1: Converting PDF to images")
        print("=" * 60)

        image_paths = pdf_to_images(pdf_path, self.config.pipeline)

        # Filter to specific page range if requested
        if question_pages:
            start_p, end_p = question_pages
            image_paths = [
                p for p in image_paths
                if start_p <= get_page_number_from_path(p) <= end_p
            ]
            print(f"Filtered to pages {start_p}-{end_p}: {len(image_paths)} pages")

        # ─── Stage 2: Vision Agent ───────────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 2: Vision Agent - Extracting questions from images")
        print("=" * 60)

        # Separate answer key pages from question pages
        answer_image_paths = []
        question_image_paths = image_paths

        if answer_key_pages:
            answer_image_paths = [
                p for p in image_paths
                if get_page_number_from_path(p) in answer_key_pages
            ]
            question_image_paths = [
                p for p in image_paths
                if get_page_number_from_path(p) not in answer_key_pages
            ]

        raw_extractions = self.vision_agent.extract_pages(question_image_paths)

        # ─── Stage 3: Structuring Agent ──────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 3: Structuring Agent - Converting to structured JSON")
        print("=" * 60)

        structured_questions = self.structuring_agent.structure_batch(raw_extractions)

        # ─── Stage 4: Answer Agent ───────────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 4: Answer Agent - Mapping correct answers")
        print("=" * 60)

        answer_key: Dict[int, str] = {}
        if answer_key_text:
            answer_key = self.answer_agent.parse_answer_key_text(answer_key_text)
        elif answer_image_paths:
            answer_key = self.answer_agent.parse_answer_key_images(answer_image_paths)

        if answer_key:
            structured_questions = self.answer_agent.merge_answers(
                structured_questions, answer_key
            )
        else:
            print("[Warning] No answer key provided. Correct answers not set.")

        # ─── Stage 5: Validator Agent ────────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 5: Validator Agent - Validating questions")
        print("=" * 60)

        validation_results = self.validator_agent.validate_all(structured_questions)
        valid_questions = validation_results["valid_questions"]

        # ─── Stage 6: Difficulty & Topic Classification ──────────────
        print("\n" + "=" * 60)
        print("STAGE 6: Difficulty classification & topic tagging")
        print("=" * 60)

        valid_questions = self.difficulty_classifier.classify_batch(valid_questions)

        # Report difficulty distribution
        diff_counts = {}
        for q in valid_questions:
            diff_counts[q.difficulty] = diff_counts.get(q.difficulty, 0) + 1
        print(f"Difficulty distribution: {diff_counts}")

        # ─── Stage 7: Export & Database ──────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 7: Exporting results")
        print("=" * 60)

        # Always export to JSON and CSV
        self._export_json(valid_questions, output_json)
        export_to_csv(valid_questions, output_csv)

        # Database insertion
        if not skip_db:
            try:
                db = DatabaseConnection(self.config.database)
                db.initialize()
                inserted = bulk_insert(db, valid_questions, batch_size=self.config.pipeline.batch_size)
                print(f"Inserted {inserted} questions into PostgreSQL.")
            except Exception as e:
                print(f"[Warning] Database insertion failed: {e}")
                print("Questions have been saved to CSV/JSON as fallback.")

        # ─── Summary ─────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total pages processed: {len(question_image_paths)}")
        print(f"Questions extracted:   {len(structured_questions)}")
        print(f"Valid questions:       {len(valid_questions)}")
        print(f"Invalid questions:     {validation_results['invalid']}")
        print(f"Time elapsed:          {elapsed:.1f}s")
        print(f"Output CSV:            {output_csv}")
        print(f"Output JSON:           {output_json}")

        # Cleanup GPU
        self.vision_agent.cleanup()

        return valid_questions

    def _export_json(self, questions: List[StructuredQuestion], output_path: str):
        """Export questions to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [q.to_dict() for q in questions]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(questions)} questions to {output_path}")
