"""
PDF to Images converter.
Converts each page of a PDF book into individual image files.
"""

import fitz  # PyMuPDF
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List

from book_parser.config import PipelineConfig


def convert_page(args: tuple) -> str:
    """Convert a single PDF page to an image. Returns the output path."""
    pdf_path, page_num, output_dir, dpi, fmt = args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    output_path = Path(output_dir) / f"page_{page_num + 1}.{fmt}"
    pix.save(str(output_path))
    doc.close()
    return str(output_path)


def pdf_to_images(
    pdf_path: str,
    config: PipelineConfig,
) -> List[str]:
    """
    Convert all pages of a PDF to images.

    Args:
        pdf_path: Path to the input PDF file.
        config: Pipeline configuration.

    Returns:
        List of output image file paths.
    """
    Path(config.pages_dir).mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    print(f"Converting {total_pages} pages to images (DPI={config.dpi})...")

    args_list = [
        (pdf_path, i, config.pages_dir, config.dpi, config.image_format)
        for i in range(total_pages)
    ]

    image_paths = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for i, path in enumerate(executor.map(convert_page, args_list)):
            image_paths.append(path)
            if (i + 1) % 50 == 0:
                print(f"  Converted {i + 1}/{total_pages} pages...")

    print(f"All {total_pages} pages converted to {config.pages_dir}/")
    return sorted(image_paths)


def get_page_number_from_path(image_path: str) -> int:
    """Extract page number from image filename like page_42.png -> 42."""
    stem = Path(image_path).stem  # page_42
    return int(stem.split("_")[1])
