# Project Context

This project builds an automated system that converts aptitude books into a structured question bank.

## Dataset

The source book contains:

* ~926 pages
* ~20–25 questions per page
* ~20,000 questions total

Questions include:

* question text
* four options
* mathematical expressions
* answer key located separately

## Technology Stack

Processing:

* Python
* LangChain

Models:

* HuggingFace
* Qwen2-VL (vision extraction)
* Mistral or Gemma (text processing)

Libraries:

* pdf2image
* PyMuPDF
* transformers

Database:

* PostgreSQL

## Output

The final dataset must contain structured questions ready for insertion into a question bank database.
