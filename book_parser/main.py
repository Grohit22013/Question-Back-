"""
Main entry point for the Book-to-Question-Bank ingestion system.

Usage:
    # Full pipeline
    python -m book_parser.main --pdf book.pdf --answer-pages 900-926

    # Process specific page range
    python -m book_parser.main --pdf book.pdf --pages 3-50 --skip-db

    # CSV-only mode (no database)
    python -m book_parser.main --pdf book.pdf --skip-db --output questions.csv
"""

import argparse
import sys

from book_parser.config import AppConfig
from book_parser.orchestrator import BookIngestionPipeline


def parse_page_range(range_str: str) -> tuple:
    """Parse a page range string like '3-50' into (3, 50)."""
    parts = range_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid page range: {range_str}. Expected format: start-end")
    return int(parts[0]), int(parts[1])


def parse_page_list(list_str: str) -> list:
    """Parse a comma-separated or range page list like '900,901,902' or '900-926'."""
    pages = []
    for part in list_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return pages


def main():
    parser = argparse.ArgumentParser(
        description="Book-to-Question-Bank Ingestion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m book_parser.main --pdf book.pdf --answer-pages 900-926
  python -m book_parser.main --pdf book.pdf --pages 3-50 --skip-db
  python -m book_parser.main --pdf book.pdf --skip-db --output-csv results.csv
        """,
    )

    parser.add_argument(
        "--pdf", required=True, help="Path to the input PDF book file."
    )
    parser.add_argument(
        "--pages", type=str, default=None,
        help="Page range to process (e.g., '3-50'). Default: all pages.",
    )
    parser.add_argument(
        "--answer-pages", type=str, default=None,
        help="Pages containing the answer key (e.g., '900-926' or '900,901,902').",
    )
    parser.add_argument(
        "--answer-text-file", type=str, default=None,
        help="Path to a text file containing the answer key.",
    )
    parser.add_argument(
        "--skip-db", action="store_true",
        help="Skip database insertion.",
    )
    parser.add_argument(
        "--output-csv", type=str, default="output/questions.csv",
        help="Path for CSV output file.",
    )
    parser.add_argument(
        "--output-json", type=str, default="output/questions.json",
        help="Path for JSON output file.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for PDF conversion.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI resolution for PDF to image conversion.",
    )
    parser.add_argument(
        "--vision-model", type=str, default=None,
        help="Override the vision model name.",
    )
    parser.add_argument(
        "--text-model", type=str, default=None,
        help="Override the text model name.",
    )
    parser.add_argument(
        "--db-host", type=str, default=None, help="PostgreSQL host."
    )
    parser.add_argument(
        "--db-name", type=str, default=None, help="PostgreSQL database name."
    )
    parser.add_argument(
        "--db-user", type=str, default=None, help="PostgreSQL user."
    )
    parser.add_argument(
        "--db-password", type=str, default=None, help="PostgreSQL password."
    )

    args = parser.parse_args()

    # Build config
    config = AppConfig()
    config.pipeline.workers = args.workers
    config.pipeline.dpi = args.dpi

    if args.vision_model:
        config.model.vision_model = args.vision_model
    if args.text_model:
        config.model.text_model = args.text_model
    if args.db_host:
        config.database.host = args.db_host
    if args.db_name:
        config.database.database = args.db_name
    if args.db_user:
        config.database.user = args.db_user
    if args.db_password:
        config.database.password = args.db_password

    # Parse page ranges
    question_pages = None
    if args.pages:
        question_pages = parse_page_range(args.pages)

    answer_key_pages = None
    if args.answer_pages:
        answer_key_pages = parse_page_list(args.answer_pages)

    answer_key_text = None
    if args.answer_text_file:
        with open(args.answer_text_file, "r", encoding="utf-8") as f:
            answer_key_text = f.read()

    # Run pipeline
    pipeline = BookIngestionPipeline(config)

    questions = pipeline.run(
        pdf_path=args.pdf,
        answer_key_pages=answer_key_pages,
        answer_key_text=answer_key_text,
        question_pages=question_pages,
        skip_db=args.skip_db,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )

    print(f"\nDone. {len(questions)} questions ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
