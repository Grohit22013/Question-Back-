"""
Vision Agent - LangChain agent wrapping the Qwen2-VL vision extractor.
Processes page images and returns raw extracted question text.
"""

from typing import List

from book_parser.config import ModelConfig
from book_parser.pipeline.vision_extractor import VisionExtractor
from book_parser.schemas import RawExtraction


class VisionAgent:
    """
    LangChain agent for vision-based question extraction.
    Wraps the Qwen2-VL model to extract questions from page images.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.extractor = VisionExtractor(config)

    def extract_page(self, image_path: str, page_number: int) -> RawExtraction:
        """
        Extract questions from a single page image.

        Args:
            image_path: Path to the page image file.
            page_number: The page number in the book.

        Returns:
            RawExtraction with the extracted text.
        """
        raw_text = self.extractor.extract_from_image(image_path)
        return RawExtraction(
            page_number=page_number,
            raw_text=raw_text,
            image_path=image_path,
        )

    def extract_pages(self, image_paths: List[str]) -> List[RawExtraction]:
        """
        Extract questions from multiple page images sequentially.

        Args:
            image_paths: List of page image file paths.

        Returns:
            List of RawExtraction objects.
        """
        from book_parser.pipeline.pdf_to_images import get_page_number_from_path

        results = []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            page_num = get_page_number_from_path(path)
            print(f"[VisionAgent] Processing page {page_num} ({i+1}/{total})...")

            extraction = self.extract_page(path, page_num)
            results.append(extraction)

            if (i + 1) % 10 == 0:
                print(f"[VisionAgent] Completed {i+1}/{total} pages")

        print(f"[VisionAgent] Finished extracting {total} pages.")
        return results

    def cleanup(self):
        """Release GPU resources."""
        self.extractor.unload_model()
