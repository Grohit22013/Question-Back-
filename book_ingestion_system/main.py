"""
Main pipeline orchestrator — Book-to-Question-Bank Ingestion System.

Executes the full pipeline as defined in masterprompt.md:
  PDF → Images → Vision Parse → Extract → Answer Keys → Merge →
  Tag Topics → Classify Difficulty → Validate → Self-Heal → JSON Output → DB Insert
"""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PDF_PATH,
    PAGES_DIR,
    OUTPUT_DIR,
    LOGS_DIR,
    BATCH_SIZE,
    PARALLEL_WORKERS,
    MAX_RETRIES,
    QUESTION_START_PAGE,
    QUESTION_END_PAGE,
    ANSWER_KEY_START_PAGE,
    ANSWER_KEY_END_PAGE,
)
from models.question_schema import RawQuestion, QuestionRecord
from pipeline.pdf_to_images import convert_pdf_to_images
from pipeline.question_extractor import extract_questions_from_page
from pipeline.answer_key_parser import load_answer_keys
from pipeline.question_answer_merger import merge_all
from pipeline.topic_tagger import assign_topics
from pipeline.difficulty_classifier import classify_difficulty
from pipeline.validator import validate_records
from pipeline.self_healing_engine import heal_invalid_records

# ─── Logging Setup ────────────────────────────────────────────────────────────

LOG_FILE = LOGS_DIR / "pipeline.log"


def setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


logger = logging.getLogger("main")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_json(data: list[dict], filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d records → %s", len(data), filepath)


def save_manual_review(
    broken: list[tuple[QuestionRecord, list[str]]],
    filepath: Path,
):
    entries = []
    for rec, errs in broken:
        entries.append({
            "question_name": rec.question_name,
            "errors": errs,
            "tags": rec.tags,
            "weightage": rec.weightage,
        })
    save_json(entries, filepath)


# ─── Pipeline Steps ──────────────────────────────────────────────────────────

def step1_convert_pdf(start: int, end: int) -> list[Path]:
    """Convert PDF pages to images."""
    logger.info("═══ STEP 1: PDF → Images ═══")
    images = convert_pdf_to_images(
        pdf_path=PDF_PATH,
        output_dir=PAGES_DIR,
        start_page=start - 1,  # fitz uses 0-based
        end_page=end,
    )
    logger.info("Step 1 complete — %d page images created.", len(images))
    return images


def step2_extract_questions(
    question_images: list[Path],
    start_page: int,
    workers: int = PARALLEL_WORKERS,
) -> list[RawQuestion]:
    """Extract questions from page images using parallel workers."""
    logger.info("═══ STEP 2: Vision Extraction ═══")
    all_questions: list[RawQuestion] = []

    def _extract(args):
        img_path, page_num = args
        return extract_questions_from_page(img_path, page_num)

    tasks = [
        (img, start_page + idx)
        for idx, img in enumerate(question_images)
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                page_questions = future.result()
                all_questions.extend(page_questions)
            except Exception as exc:
                img, pg = futures[future]
                logger.error("Extraction failed for page %d (%s): %s", pg, img.name, exc)

    logger.info("Step 2 complete — %d questions extracted.", len(all_questions))
    return all_questions


def step3_parse_answer_keys(answer_images: list[Path]) -> dict[str, str]:
    """Parse answer key pages."""
    logger.info("═══ STEP 3: Answer Key Parsing ═══")
    keys = load_answer_keys(image_paths=answer_images)
    logger.info("Step 3 complete — %d answer keys loaded.", len(keys))
    return keys


def step4_merge(
    raw_questions: list[RawQuestion],
    answer_keys: dict[str, str],
) -> list[QuestionRecord]:
    """Merge questions with answers."""
    logger.info("═══ STEP 4: Merge Questions + Answers ═══")
    records = merge_all(raw_questions, answer_keys)
    logger.info("Step 4 complete — %d records merged.", len(records))
    return records


def step5_tag_topics(
    records: list[QuestionRecord],
    raw_questions: list[RawQuestion],
) -> list[QuestionRecord]:
    """Assign topic tags."""
    logger.info("═══ STEP 5: Topic Tagging ═══")
    records = assign_topics(records, raw_questions)
    logger.info("Step 5 complete.")
    return records


def step6_classify_difficulty(
    records: list[QuestionRecord],
    use_llm: bool = False,
) -> list[QuestionRecord]:
    """Classify difficulty levels."""
    logger.info("═══ STEP 6: Difficulty Classification ═══")
    records = classify_difficulty(records, use_llm=use_llm)
    logger.info("Step 6 complete.")
    return records


def step7_validate(records: list[QuestionRecord]):
    """Validate all records."""
    logger.info("═══ STEP 7: Validation ═══")
    result = validate_records(records)
    logger.info(
        "Step 7 complete — %d valid, %d invalid.",
        len(result.valid), len(result.invalid),
    )
    return result


def step8_self_heal(
    invalid_records,
    page_images_map: dict[int, Path],
    raw_questions: list[RawQuestion],
    answer_keys: dict[str, str],
):
    """Self-healing error correction."""
    logger.info("═══ STEP 8: Self-Healing ═══")
    healed, still_broken = heal_invalid_records(
        invalid_records,
        page_images_map,
        raw_questions,
        answer_keys,
        max_retries=MAX_RETRIES,
    )
    logger.info(
        "Step 8 complete — healed: %d, flagged for review: %d.",
        len(healed), len(still_broken),
    )
    return healed, still_broken


def step9_save_output(records: list[QuestionRecord]):
    """Save final structured JSON."""
    logger.info("═══ STEP 9: Save Output ═══")
    data = [r.model_dump() for r in records]
    save_json(data, OUTPUT_DIR / "question_bank.json")
    logger.info("Step 9 complete — %d records saved.", len(records))


def step10_db_insert(records: list[QuestionRecord], skip_db: bool = False):
    """Bulk insert into PostgreSQL."""
    logger.info("═══ STEP 10: Database Insertion ═══")
    if skip_db:
        logger.info("Database insertion skipped (--skip-db flag).")
        return

    try:
        from database.connection import init_pool, ensure_table, close_pool
        from database.bulk_insert import bulk_insert

        init_pool()
        ensure_table()
        inserted = bulk_insert(records, batch_size=BATCH_SIZE)
        logger.info("Step 10 complete — %d records inserted.", inserted)
        close_pool()
    except Exception as exc:
        logger.error("Database insertion failed: %s", exc)
        logger.info("JSON output is still available at %s", OUTPUT_DIR / "question_bank.json")


# ─── Main Orchestrator ───────────────────────────────────────────────────────

def run_pipeline(
    skip_db: bool = False,
    use_llm_difficulty: bool = False,
    skip_pdf_convert: bool = False,
):
    """
    Execute the full ingestion pipeline end-to-end.
    """
    setup_logging()
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  BOOK-TO-QUESTION-BANK INGESTION PIPELINE")
    logger.info("  PDF: %s", PDF_PATH)
    logger.info("=" * 60)

    # ── Step 1: PDF → Images ──
    if skip_pdf_convert:
        logger.info("Skipping PDF conversion — using existing images.")
        question_images = sorted(PAGES_DIR.glob("page_*.png"))
    else:
        all_images = step1_convert_pdf(QUESTION_START_PAGE, QUESTION_END_PAGE)
        # Split question pages vs answer key pages
        question_images = all_images
        # If answer key pages are defined, convert them too
        answer_images = []
        if ANSWER_KEY_START_PAGE > 0 and ANSWER_KEY_END_PAGE > 0:
            answer_images = step1_convert_pdf(ANSWER_KEY_START_PAGE, ANSWER_KEY_END_PAGE)

    # Build page_number → image_path map
    page_images_map: dict[int, Path] = {}
    for img in question_images:
        page_num = int(img.stem.split("_")[1])
        page_images_map[page_num] = img

    # ── Step 2: Extract questions ──
    raw_questions = step2_extract_questions(
        question_images,
        start_page=QUESTION_START_PAGE,
        workers=PARALLEL_WORKERS,
    )

    # ── Step 3: Parse answer keys ──
    answer_images_list = []
    if ANSWER_KEY_START_PAGE > 0 and ANSWER_KEY_END_PAGE > 0:
        answer_images_list = sorted(
            PAGES_DIR.glob("page_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        answer_images_list = [
            p for p in answer_images_list
            if ANSWER_KEY_START_PAGE <= int(p.stem.split("_")[1]) <= ANSWER_KEY_END_PAGE
        ]
    answer_keys = step3_parse_answer_keys(answer_images_list)

    # ── Step 4: Merge ──
    records = step4_merge(raw_questions, answer_keys)

    # ── Step 5: Topic tags ──
    records = step5_tag_topics(records, raw_questions)

    # ── Step 6: Difficulty ──
    records = step6_classify_difficulty(records, use_llm=use_llm_difficulty)

    # ── Step 7: Validate ──
    validation = step7_validate(records)

    # ── Step 8: Self-heal ──
    healed: list[QuestionRecord] = []
    still_broken = []
    if validation.invalid:
        healed, still_broken = step8_self_heal(
            validation.invalid,
            page_images_map,
            raw_questions,
            answer_keys,
        )

    # Combine valid + healed
    final_records = validation.valid + healed

    # Re-validate healed records
    if healed:
        healed_validation = validate_records(healed)
        final_records = validation.valid + healed_validation.valid
        still_broken.extend(healed_validation.invalid)

    # ── Step 9: Save JSON ──
    step9_save_output(final_records)

    # Save manual review file
    if still_broken:
        save_manual_review(still_broken, OUTPUT_DIR / "manual_review.json")

    # Save validation report
    report = {
        "total_extracted": len(raw_questions),
        "total_valid": len(final_records),
        "total_invalid": len(still_broken),
        "answer_keys_loaded": len(answer_keys),
    }
    save_json([report], OUTPUT_DIR / "validation_report.json")

    # ── Step 10: DB insert ──
    step10_db_insert(final_records, skip_db=skip_db)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Total questions: %d", len(final_records))
    logger.info("  Flagged for review: %d", len(still_broken))
    logger.info("  Elapsed: %.1f seconds", elapsed)
    logger.info("=" * 60)

    return final_records


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Book-to-Question-Bank Ingestion Pipeline")
    parser.add_argument("--skip-db", action="store_true", help="Skip database insertion")
    parser.add_argument("--skip-pdf-convert", action="store_true", help="Use existing page images")
    parser.add_argument("--llm-difficulty", action="store_true", help="Use LLM for difficulty classification")
    args = parser.parse_args()

    run_pipeline(
        skip_db=args.skip_db,
        use_llm_difficulty=args.llm_difficulty,
        skip_pdf_convert=args.skip_pdf_convert,
    )
