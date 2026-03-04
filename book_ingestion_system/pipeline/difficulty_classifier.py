"""
Step 6 — Difficulty classifier.

Classifies each question as Easy, Medium, or Hard using heuristics
or an LLM call.
"""

import json
import logging

from models.question_schema import QuestionRecord
from config import VISION_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)


def _heuristic_difficulty(question_text: str) -> str:
    """
    Simple rule-based difficulty classifier.
    Falls back to this when LLM classification is unavailable or too slow.
    """
    text = question_text.lower()

    hard_indicators = [
        "probability", "permutation", "combination", "integral",
        "derivative", "log", "trigonometr", "√", "³√",
        "simultaneous", "quadratic equation",
    ]
    medium_indicators = [
        "ratio", "proportion", "percentage", "profit", "loss",
        "compound interest", "average", "speed", "distance",
        "time and work", "pipes", "cistern",
    ]

    # Check length and nesting as complexity signals
    word_count = len(text.split())

    if any(ind in text for ind in hard_indicators) or word_count > 60:
        return "hard"
    if any(ind in text for ind in medium_indicators) or word_count > 35:
        return "medium"
    return "easy"


def _llm_classify_batch(questions: list[str]) -> list[str]:
    """
    Use the LLM to classify a batch of questions by difficulty.
    Returns a list of difficulty labels matching the input order.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        logger.warning("OpenAI not available for difficulty classification — using heuristics")
        return [_heuristic_difficulty(q) for q in questions]

    prompt = (
        "Classify each of the following aptitude questions as exactly one of: easy, medium, hard.\n"
        "Consider: number of steps, mathematical complexity, and conceptual difficulty.\n"
        "Return ONLY a JSON array of strings in the same order. Example: [\"easy\", \"medium\", \"hard\"]\n\n"
    )
    for i, q in enumerate(questions):
        prompt += f"{i + 1}. {q}\n"

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        labels = json.loads(raw)
        if isinstance(labels, list) and len(labels) == len(questions):
            return [l.lower() if l.lower() in {"easy", "medium", "hard"} else "medium" for l in labels]
    except Exception as exc:
        logger.warning("LLM difficulty classification failed: %s — falling back to heuristics", exc)

    return [_heuristic_difficulty(q) for q in questions]


def classify_difficulty(
    records: list[QuestionRecord],
    use_llm: bool = False,
    batch_size: int = 50,
) -> list[QuestionRecord]:
    """
    Assign difficulty (weightage) to each QuestionRecord.
    """
    if use_llm:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            texts = [r.question_name for r in batch]
            labels = _llm_classify_batch(texts)
            for rec, label in zip(batch, labels):
                rec.weightage = label
    else:
        for rec in records:
            rec.weightage = _heuristic_difficulty(rec.question_name)

    dist: dict[str, int] = {}
    for r in records:
        dist[r.weightage] = dist.get(r.weightage, 0) + 1
    logger.info("Difficulty classification complete — distribution: %s", dist)

    return records
