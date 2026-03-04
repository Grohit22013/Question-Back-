"""
Step 1 — Convert PDF pages to images using PyMuPDF (fitz).
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

from config import PDF_PATH, PAGES_DIR, DPI

logger = logging.getLogger(__name__)


def convert_pdf_to_images(
    pdf_path: Path = PDF_PATH,
    output_dir: Path = PAGES_DIR,
    dpi: int = DPI,
    start_page: int = 0,
    end_page: int | None = None,
) -> list[Path]:
    """
    Convert each page of the PDF to a PNG image.

    Returns a list of image file paths in page order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    end = end_page if end_page is not None else total_pages

    image_paths: list[Path] = []
    zoom = dpi / 72  # fitz default is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(start_page, min(end, total_pages)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        img_path = output_dir / f"page_{page_num + 1}.png"
        pix.save(str(img_path))
        image_paths.append(img_path)

        if (page_num + 1) % 50 == 0 or page_num == start_page:
            logger.info("Converted page %d / %d", page_num + 1, total_pages)

    doc.close()
    logger.info("PDF conversion complete — %d images saved to %s", len(image_paths), output_dir)
    return image_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = convert_pdf_to_images()
    print(f"Converted {len(paths)} pages.")
