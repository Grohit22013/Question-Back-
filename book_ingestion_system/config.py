"""
Configuration for the Book-to-Question-Bank Ingestion Pipeline.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PDF_PATH = PROJECT_ROOT / "R S Agarwal Chat.pdf"
PAGES_DIR = BASE_DIR / "pages"
LOGS_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
PAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Vision LLM Configuration ────────────────────────────────────────────────
# Supported providers: "openai", "gemini", "anthropic", "huggingface"
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "huggingface")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# ─── Hugging Face Local Model ─────────────────────────────────────────────────
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

# ─── Database Configuration ──────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "question_bank")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

# ─── Pipeline Settings ───────────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
DPI = int(os.getenv("PDF_DPI", "200"))

# ─── Topic Mapping (page ranges → topic tag) ─────────────────────────────────
# Adjust these ranges based on the actual book index.
TOPIC_MAP = [
    (3, 50, "number_system"),
    (51, 68, "hcf_lcm"),
    (69, 94, "decimal_fractions"),
    (95, 179, "simplification"),
    (180, 210, "average"),
    (211, 250, "problems_on_ages"),
    (251, 290, "percentage"),
    (291, 330, "profit_loss"),
    (331, 370, "ratio_proportion"),
    (371, 410, "partnership"),
    (411, 450, "time_work"),
    (451, 500, "pipes_cisterns"),
    (501, 550, "time_distance"),
    (551, 600, "trains"),
    (601, 640, "boats_streams"),
    (641, 680, "simple_interest"),
    (681, 720, "compound_interest"),
    (721, 760, "area"),
    (761, 800, "volume_surface_area"),
    (801, 840, "permutations_combinations"),
    (841, 880, "probability"),
    (881, 926, "miscellaneous"),
]

# ─── Answer Key Page Ranges ──────────────────────────────────────────────────
# Pages that contain answer keys (adjust after inspecting the PDF).
ANSWER_KEY_START_PAGE = int(os.getenv("ANSWER_KEY_START_PAGE", "0"))
ANSWER_KEY_END_PAGE = int(os.getenv("ANSWER_KEY_END_PAGE", "0"))

# ─── Question Pages (pages containing actual questions) ──────────────────────
QUESTION_START_PAGE = int(os.getenv("QUESTION_START_PAGE", "1"))
QUESTION_END_PAGE = int(os.getenv("QUESTION_END_PAGE", "926"))
