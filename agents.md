# LangChain Multi-Agent System

The system uses multiple specialized agents.

## Vision Agent

Purpose:
Extract questions from page images.

Input:
Page image

Output:
Raw question text with options

Model:
Qwen2-VL

---

## Structuring Agent

Purpose:
Convert raw text into structured JSON.

Tasks:

* clean question text
* identify options
* preserve equations

---

## Answer Agent

Purpose:
Parse answer key pages and map answers.

Input:
Answer key text

Example:

331 (c)
332 (b)

Output:

{
"331": "c",
"332": "b"
}

---

## Validator Agent

Purpose:
Validate and repair extracted questions.

Checks:

* question text exists
* four options exist
* exactly one correct answer
