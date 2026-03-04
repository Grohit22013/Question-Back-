"""
Vision Extractor - Uses Qwen2-VL via HuggingFace transformers to extract
question text from page images.
"""

import torch
from pathlib import Path
from typing import Optional

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from book_parser.config import ModelConfig


class VisionExtractor:
    """Extracts raw text from page images using Qwen2-VL."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the Qwen2-VL model and processor."""
        if self.model is not None:
            return

        print(f"Loading vision model: {self.config.vision_model}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.vision_model,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.vision_model
        )
        print("Vision model loaded.")

    def extract_from_image(self, image_path: str) -> str:
        """
        Extract raw question text from a single page image.

        Args:
            image_path: Path to the page image.

        Returns:
            Raw extracted text containing questions and options.
        """
        self.load_model()

        image_uri = Path(image_path).resolve().as_uri()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL multiple choice questions from this page image.\n\n"
                            "Rules:\n"
                            "1. Preserve mathematical expressions exactly as shown.\n"
                            "2. Convert superscripts like ² to ^2 notation.\n"
                            "3. Convert square root symbols to sqrt().\n"
                            "4. Include the question number, question text, and all four options (a, b, c, d).\n"
                            "5. Ignore page headers, footers, and page numbers.\n"
                            "6. Output each question in this format:\n\n"
                            "Q<number>: <question text>\n"
                            "(a) <option a>\n"
                            "(b) <option b>\n"
                            "(c) <option c>\n"
                            "(d) <option d>\n\n"
                            "Extract every question visible on the page."
                        ),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else ""

    def unload_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
