# System Architecture

The system processes a book through a multi-stage pipeline.

## Pipeline

PDF Book
↓
Convert pages to images
↓
Vision Agent (extract questions)
↓
Structuring Agent
↓
Answer Key Agent
↓
Validator Agent
↓
Topic Tagging
↓
Difficulty Classification
↓
Structured JSON
↓
Database Insert

## Project Structure

book_parser/

pipeline/

* pdf_to_images.py
* vision_extractor.py
* question_parser.py
* answer_parser.py
* validator.py
* difficulty_classifier.py

agents/

* vision_agent.py
* structuring_agent.py
* answer_agent.py
* validator_agent.py

database/

* connection.py
* bulk_insert.py
