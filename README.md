# Book-to-Question-Bank Ingestion System

A **LangChain multi-agent** system powered by **HuggingFace** models that converts aptitude books (PDFs) into a structured question bank database.

## Architecture

```
PDF Book
  ↓
Convert pages to images (PyMuPDF)
  ↓
Vision Agent (Qwen2-VL) → Extract questions from images
  ↓
Structuring Agent (Mistral) → Convert to structured JSON
  ↓
Answer Key Agent → Parse & merge correct answers
  ↓
Validator Agent → Validate & repair questions
  ↓
Difficulty Classification + Topic Tagging
  ↓
PostgreSQL Database / CSV / JSON export
```

## Tech Stack

| Component        | Technology                     |
| ---------------- | ------------------------------ |
| Orchestration    | LangChain                      |
| Vision Model     | Qwen2-VL (HuggingFace)        |
| Text Model       | Mistral-7B (HuggingFace)      |
| PDF Processing   | PyMuPDF                        |
| Database         | PostgreSQL                     |
| Language         | Python 3.10+                   |

## Project Structure

```
book_parser/
├── __init__.py
├── config.py              # All configuration (models, DB, topics)
├── schemas.py             # Data models (StructuredQuestion, etc.)
├── orchestrator.py        # Main LangChain multi-agent pipeline
├── main.py                # CLI entry point
├── pipeline/
│   ├── pdf_to_images.py       # PDF → page images
│   ├── vision_extractor.py    # Qwen2-VL image extraction
│   ├── question_parser.py     # LLM-based text → JSON structuring
│   ├── answer_parser.py       # Answer key parsing
│   ├── validator.py           # Question validation & repair
│   └── difficulty_classifier.py  # Difficulty classification
├── agents/
│   ├── vision_agent.py        # Vision extraction agent
│   ├── structuring_agent.py   # Text structuring agent
│   ├── answer_agent.py        # Answer key parsing agent
│   └── validator_agent.py     # Validation agent
└── database/
    ├── connection.py          # PostgreSQL connection manager
    └── bulk_insert.py         # Batch insert & CSV export
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up PostgreSQL (optional)

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=question_bank
export DB_USER=postgres
export DB_PASSWORD=your_password
```

### 3. GPU Requirements

- **Qwen2-VL-7B**: ~16GB VRAM recommended
- **Mistral-7B**: ~14GB VRAM recommended
- For smaller GPUs, use quantized models via `--vision-model` and `--text-model` flags

## Usage

### Full pipeline (PDF → Database)

```bash
python -m book_parser.main --pdf "R S Agarwal Chat.pdf" --answer-pages 900-926
```

### Process specific pages only

```bash
python -m book_parser.main --pdf book.pdf --pages 3-50 --skip-db
```

### Skip database, export CSV/JSON only

```bash
python -m book_parser.main --pdf book.pdf --skip-db \
  --output-csv output/questions.csv \
  --output-json output/questions.json
```

### Use custom models

```bash
python -m book_parser.main --pdf book.pdf \
  --vision-model "Qwen/Qwen2-VL-2B-Instruct" \
  --text-model "google/gemma-2b-it" \
  --skip-db
```

### Provide answer key as text file

```bash
python -m book_parser.main --pdf book.pdf \
  --answer-text-file answers.txt \
  --skip-db
```

## Output Schema

Each question follows this structure:

```json
{
  "question_name": "What is the sum of first 100 natural numbers?",
  "question_type_id": 1,
  "tags": "number_system",
  "weightage": "2",
  "is_active": true,
  "choice1_text": "5050",
  "choice1_isCorrect": true,
  "choice2_text": "5000",
  "choice2_isCorrect": false,
  "choice3_text": "5100",
  "choice3_isCorrect": false,
  "choice4_text": "4950",
  "choice4_isCorrect": false
}
```

## Pipeline Agents

| Agent              | Model     | Purpose                              |
| ------------------ | --------- | ------------------------------------ |
| Vision Agent       | Qwen2-VL  | Extract questions from page images   |
| Structuring Agent  | Mistral   | Convert raw text → structured JSON   |
| Answer Agent       | Mistral   | Parse answer keys, merge answers     |
| Validator Agent    | Rule-based| Validate schema, repair issues       |

## Validation Rules

1. Question text must exist
2. Exactly 4 options must exist
3. Exactly 1 correct answer must be marked
4. Question numbers must be unique
5. Mathematical expressions must remain intact

Questions that fail validation are attempted for repair. Unrepairable questions are flagged for manual review.

## Topic Mapping

Topics are auto-assigned based on page ranges (configurable in `config.py`):

| Pages   | Topic                    |
| ------- | ------------------------ |
| 3-50    | number_system            |
| 51-100  | hcf_lcm                 |
| 101-150 | fractions_decimals       |
| ...     | *(see config.py)*        |
