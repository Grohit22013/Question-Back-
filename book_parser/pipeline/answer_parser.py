"""
Answer Parser - Extracts answer key mappings from answer key pages.
Uses HuggingFace model via LangChain to parse answer key pages.
"""

import json
import re
from typing import Dict

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline as hf_pipeline

from book_parser.config import ModelConfig
from book_parser.schemas import AnswerMapping


ANSWER_KEY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a data extraction assistant. "
        "Extract answer key mappings from the given text. "
        "Return ONLY a JSON object mapping question numbers to correct answers.",
    ),
    (
        "human",
        "Extract the answer key from this text. Each entry maps a question number to its "
        "correct option letter (a, b, c, or d).\n\n"
        "Text:\n{answer_text}\n\n"
        "Return a JSON object like:\n"
        '{{"1": "c", "2": "b", "3": "a"}}',
    ),
])


class AnswerParser:
    """Parses answer key pages to extract question-to-answer mappings."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.chain = None

    def _build_chain(self):
        if self.chain is not None:
            return
        pipe = hf_pipeline(
            "text-generation",
            model=self.config.text_model,
            max_new_tokens=2048,
            do_sample=False,
            return_full_text=False,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        self.chain = ANSWER_KEY_PROMPT | llm | StrOutputParser()

    def parse_answer_key(self, answer_text: str) -> Dict[int, str]:
        """
        Parse answer key text into a mapping using the LLM chain.

        Args:
            answer_text: Raw text from answer key pages.

        Returns:
            Dict mapping question_number -> correct_option ("a","b","c","d").
        """
        self._build_chain()
        response = self.chain.invoke({"answer_text": answer_text})
        return self._parse_response(response)

    def parse_answer_key_regex(self, answer_text: str) -> Dict[int, str]:
        """
        Regex-based fallback parser for answer keys.

        Handles formats like:
            331 (c)
            332 (b)
            333. (a)
            334 - c
        """
        answers = {}
        # Pattern: number followed by answer letter in various formats
        pattern = re.compile(
            r'(\d+)\s*[\.\)\-]?\s*\(?\s*([a-dA-D])\s*\)?'
        )
        for match in pattern.finditer(answer_text):
            q_num = int(match.group(1))
            answer = match.group(2).lower()
            answers[q_num] = answer

        return answers

    def _parse_response(self, response: str) -> Dict[int, str]:
        """Parse LLM response into answer mapping dict."""
        cleaned = re.sub(r"```json\s*", "", response)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not match:
            return {}

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}

        return {int(k): v.lower() for k, v in data.items() if v.lower() in ("a", "b", "c", "d")}
