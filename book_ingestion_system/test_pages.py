"""
Test script — Process PDF pages and extract questions as JSON.
Usage:
    cd book_ingestion_system
    python test_pages.py
"""
import json
import logging
import sys
import time
import os
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# ───────── ENV LOADING ─────────
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✔ Loaded env from {env_file}")
except ImportError:
    print("⚠ python-dotenv not installed — using system env vars")
# Force Gemini provider
os.environ["VISION_PROVIDER"] = "gemini"
print("✔ Vision provider:", os.getenv("VISION_PROVIDER"))
# ───────── PROJECT IMPORTS ─────────
from config import PAGES_DIR, OUTPUT_DIR
from pipeline.pdf_to_images import convert_pdf_to_images
from pipeline.vision_page_parser import parse_page_questions
# ───────── SETTINGS ─────────
TEST_PDF = Path(__file__).resolve().parent.parent / "testdoc.pdf"
TEST_START_PAGE = 1
TEST_END_PAGE = 1
TEST_OUTPUT = OUTPUT_DIR / "test_pages_output.json"
MAX_RETRIES = 6
INITIAL_WAIT = 20
PAGE_DELAY = 25
# ───────── LOGGING ─────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_pages")

# ───────── RETRY WRAPPER ─────────
def extract_questions_with_retry(image_path):
    wait_time = INITIAL_WAIT
    for attempt in range(MAX_RETRIES):
        try:
            return parse_page_questions(image_path)
        except Exception as exc:
            if "429" in str(exc):
                logger.warning(
                    "⚠ Rate limit hit. Waiting %s seconds before retry...",
                    wait_time,
                )
                time.sleep(wait_time)
                wait_time *= 2
            else:
                raise exc
    raise RuntimeError("Max retries exceeded")

# ───────── MAIN PIPELINE ─────────
def main():
    start_time = time.time()
    logger.info(
        "Converting pages %d–%d of %s to images...",
        TEST_START_PAGE,
        TEST_END_PAGE,
        TEST_PDF.name,
    )
    if not TEST_PDF.exists():
        logger.error("PDF not found at %s", TEST_PDF)
        sys.exit(1)
    images = convert_pdf_to_images(
        pdf_path=TEST_PDF,
        output_dir=PAGES_DIR,
        start_page=TEST_START_PAGE - 1,
        end_page=TEST_END_PAGE,
    )
    logger.info("Created %d page images.", len(images))
    all_questions = []
    for img_path in images:
        page_num = int(img_path.stem.split("_")[1])
        logger.info("Extracting questions from page %d...", page_num)
        try:
            questions = extract_questions_with_retry(img_path)
            for q in questions:
                q["page_number"] = page_num
            logger.info("→ %d questions found", len(questions))
            all_questions.extend(questions)
            logger.info("Cooling down %s seconds...", PAGE_DELAY)
            time.sleep(PAGE_DELAY)
        except Exception as exc:
            logger.error("✖ Failed on page %d: %s", page_num, exc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print(f"Pages processed : {TEST_START_PAGE}-{TEST_END_PAGE}")
    print(f"Questions found : {len(all_questions)}")
    print(f"Output file     : {TEST_OUTPUT}")
    print(f"Time elapsed    : {elapsed:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
 