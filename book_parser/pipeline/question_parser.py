"""
Question Parser - Parses raw extracted text into structured question objects.
Uses a HuggingFace text model via LangChain for reliable JSON structuring.
"""

import json
import re
from typing import List

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline as hf_pipeline

from book_parser.config import ModelConfig
from book_parser.schemas import ParsedQuestion


STRUCTURING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise data extraction assistant. "
        "Convert raw question text into structured JSON. "
        "Preserve all mathematical expressions exactly. "
        "Return ONLY a JSON array, no other text.",
    ),
    (
        "human",
        "Convert the following raw extracted questions into structured JSON.\n\n"
        "Rules:\n"
        "1. Each question must have: question_number, question_text, choice_a, choice_b, choice_c, choice_d\n"
        "2. Preserve mathematical expressions (use ^2 for powers, sqrt() for roots)\n"
        "3. Clean up OCR artifacts but keep the content intact\n"
        "4. If a question number is missing, infer it from context\n\n"
        "Raw text:\n{raw_text}\n\n"
        "Return a JSON array:\n"
        '[{{"question_number": 1, "question_text": "...", '
        '"choice_a": "...", "choice_b": "...", "choice_c": "...", "choice_d": "..."}}]',
    ),
])


class QuestionParser:
    """Parses raw extracted text into structured ParsedQuestion objects."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.chain = None

    def _build_chain(self):
        """Build the LangChain pipeline with HuggingFace model."""
        if self.chain is not None:
            return

        print(f"Loading text model: {self.config.text_model}...")
        pipe = hf_pipeline(
            "text-generation",
            model=self.config.text_model,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        self.chain = STRUCTURING_PROMPT | llm | StrOutputParser()
        print("Text model loaded for structuring.")

    def parse(self, raw_text: str, page_number: int) -> List[ParsedQuestion]:
        """
        Parse raw text from a page into structured questions.

        Args:
            raw_text: Raw text extracted by the vision agent.
            page_number: Source page number.

        Returns:
            List of ParsedQuestion objects.
        """
        self._build_chain()

        response = self.chain.invoke({"raw_text": raw_text})

        return self._parse_response(response, page_number)

    def parse_with_regex(self, raw_text: str, page_number: int) -> List[ParsedQuestion]:
        """
        Fallback regex-based parser for when the LLM output is unreliable.
        Parses the Q<num>: format directly.
        """
        questions = []
        # Match pattern: Q<number>: <text> followed by (a)...(d)...
        pattern = re.compile(
            r'Q(\d+)[:\.]?\s*(.*?)\n'
            r'\(a\)\s*(.*?)\n'
            r'\(b\)\s*(.*?)\n'
            r'\(c\)\s*(.*?)\n'
            r'\(d\)\s*(.*?)(?:\n|$)',
            re.DOTALL
        )

        for match in pattern.finditer(raw_text):
            q = ParsedQuestion(
                question_number=int(match.group(1)),
                question_text=match.group(2).strip(),
                choice1=match.group(3).strip(),
                choice2=match.group(4).strip(),
                choice3=match.group(5).strip(),
                choice4=match.group(6).strip(),
                page_number=page_number,
            )
            questions.append(q)

        return questions

    def _parse_response(self, response: str, page_number: int) -> List[ParsedQuestion]:
        """Parse the LLM JSON response into ParsedQuestion objects."""
        # Clean markdown code fences
        cleaned = re.sub(r"```json\s*", "", response)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        # Extract JSON array
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []

        questions = []
        for item in data:
            if not isinstance(item, dict):
                continue
            q = ParsedQuestion(
                question_number=item.get("question_number"),
                question_text=item.get("question_text", ""),
                choice1=item.get("choice_a", ""),
                choice2=item.get("choice_b", ""),
                choice3=item.get("choice_c", ""),
                choice4=item.get("choice_d", ""),
                page_number=page_number,
            )
            questions.append(q)

        return questions
