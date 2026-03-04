"""
Difficulty Classifier - Classifies questions into Easy, Medium, Hard.
Uses heuristics and optionally an LLM for classification.
"""

from typing import List

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline as hf_pipeline

from book_parser.config import ModelConfig
from book_parser.schemas import StructuredQuestion, Difficulty


DIFFICULTY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a question difficulty classifier for aptitude/math questions. "
        "Classify each question as exactly one of: easy, medium, hard. "
        "Return ONLY the difficulty level word, nothing else.",
    ),
    (
        "human",
        "Classify the difficulty of this aptitude question:\n\n"
        "Question: {question_text}\n"
        "Options:\n"
        "(a) {choice_a}\n"
        "(b) {choice_b}\n"
        "(c) {choice_c}\n"
        "(d) {choice_d}\n\n"
        "Difficulty (easy/medium/hard):",
    ),
])


class DifficultyClassifier:
    """Classifies question difficulty using heuristics and/or LLM."""

    def __init__(self, config: ModelConfig, use_llm: bool = False):
        self.config = config
        self.use_llm = use_llm
        self.chain = None

    def _build_chain(self):
        if self.chain is not None or not self.use_llm:
            return
        pipe = hf_pipeline(
            "text-generation",
            model=self.config.text_model,
            max_new_tokens=10,
            do_sample=False,
            return_full_text=False,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        self.chain = DIFFICULTY_PROMPT | llm | StrOutputParser()

    def classify(self, question: StructuredQuestion) -> str:
        """
        Classify a single question's difficulty.

        Uses heuristic-based classification by default.
        Set use_llm=True for LLM-based classification.
        """
        if self.use_llm:
            return self._classify_with_llm(question)
        return self._classify_heuristic(question)

    def classify_batch(self, questions: List[StructuredQuestion]) -> List[StructuredQuestion]:
        """Classify difficulty for a batch of questions."""
        for q in questions:
            q.difficulty = self.classify(q)
            q.weightage = self._difficulty_to_weightage(q.difficulty)
        return questions

    def _classify_heuristic(self, question: StructuredQuestion) -> str:
        """
        Heuristic difficulty classification based on question characteristics.

        Indicators of higher difficulty:
        - Longer question text
        - Mathematical expressions (^, sqrt, fractions)
        - Multiple operations mentioned
        - Complex number patterns
        """
        text = question.question_name.lower()
        score = 0

        # Length-based scoring
        if len(text) > 200:
            score += 2
        elif len(text) > 100:
            score += 1

        # Math complexity indicators
        math_indicators = ["^", "sqrt", "/", "×", "ratio", "proportion", "log", "percentage"]
        for indicator in math_indicators:
            if indicator in text:
                score += 1

        # Multi-step problem indicators
        step_indicators = ["find the value", "if.*then", "how many", "what is the", "probability"]
        for indicator in step_indicators:
            import re
            if re.search(indicator, text):
                score += 1

        # Classify based on score
        if score <= 1:
            return Difficulty.EASY
        elif score <= 3:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD

    def _classify_with_llm(self, question: StructuredQuestion) -> str:
        """Classify using the LLM chain."""
        self._build_chain()
        result = self.chain.invoke({
            "question_text": question.question_name,
            "choice_a": question.choice1_text,
            "choice_b": question.choice2_text,
            "choice_c": question.choice3_text,
            "choice_d": question.choice4_text,
        })

        result_lower = result.strip().lower()
        if "easy" in result_lower:
            return Difficulty.EASY
        elif "hard" in result_lower:
            return Difficulty.HARD
        return Difficulty.MEDIUM

    def _difficulty_to_weightage(self, difficulty: str) -> str:
        """Map difficulty to weightage score."""
        mapping = {
            Difficulty.EASY: "1",
            Difficulty.MEDIUM: "2",
            Difficulty.HARD: "3",
        }
        return mapping.get(difficulty, "1")
