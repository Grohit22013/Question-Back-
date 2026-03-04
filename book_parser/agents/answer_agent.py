"""
Answer Agent - LangChain agent that parses answer key pages
and maps correct answers to questions.
"""

from typing import List, Dict

from book_parser.config import ModelConfig
from book_parser.pipeline.answer_parser import AnswerParser
from book_parser.pipeline.vision_extractor import VisionExtractor
from book_parser.schemas import StructuredQuestion, RawExtraction


class AnswerAgent:
    """
    Agent that processes answer key pages and merges correct answers
    with the extracted questions.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.answer_parser = AnswerParser(config)
        self.vision_extractor = VisionExtractor(config)

    def parse_answer_key_text(self, answer_text: str) -> Dict[int, str]:
        """
        Parse answer key from raw text.

        Tries regex first (fast, reliable for standard formats),
        falls back to LLM if regex yields few results.
        """
        # Try regex first
        answers = self.answer_parser.parse_answer_key_regex(answer_text)

        # If regex found very few, try LLM
        if len(answers) < 5:
            llm_answers = self.answer_parser.parse_answer_key(answer_text)
            # Merge: LLM results fill gaps
            for k, v in llm_answers.items():
                if k not in answers:
                    answers[k] = v

        return answers

    def parse_answer_key_images(self, image_paths: List[str]) -> Dict[int, str]:
        """
        Extract and parse answer keys from answer key page images.

        Args:
            image_paths: Paths to answer key page images.

        Returns:
            Dict mapping question numbers to correct answers.
        """
        all_answers = {}

        for path in image_paths:
            print(f"[AnswerAgent] Processing answer key: {path}")

            # Extract text from image
            raw_text = self.vision_extractor.extract_from_image(path)

            # Parse answers from text
            answers = self.parse_answer_key_text(raw_text)
            all_answers.update(answers)
            print(f"  -> Found {len(answers)} answers")

        print(f"[AnswerAgent] Total answers extracted: {len(all_answers)}")
        return all_answers

    def merge_answers(
        self,
        questions: List[StructuredQuestion],
        answer_key: Dict[int, str],
    ) -> List[StructuredQuestion]:
        """
        Merge answer key with structured questions.

        Maps the correct option letter to the corresponding choice flag.

        Args:
            questions: List of structured questions.
            answer_key: Dict of question_number -> correct_option ("a","b","c","d").

        Returns:
            Questions with correct answer flags set.
        """
        option_map = {"a": 1, "b": 2, "c": 3, "d": 4}
        matched = 0

        for q in questions:
            if q.question_number is None:
                continue

            correct = answer_key.get(q.question_number)
            if correct is None:
                continue

            # Reset all flags
            q.choice1_isCorrect = False
            q.choice2_isCorrect = False
            q.choice3_isCorrect = False
            q.choice4_isCorrect = False

            # Set the correct one
            choice_num = option_map.get(correct.lower())
            if choice_num == 1:
                q.choice1_isCorrect = True
            elif choice_num == 2:
                q.choice2_isCorrect = True
            elif choice_num == 3:
                q.choice3_isCorrect = True
            elif choice_num == 4:
                q.choice4_isCorrect = True

            matched += 1

        print(f"[AnswerAgent] Matched {matched}/{len(questions)} questions with answers.")
        return questions
