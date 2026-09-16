"""Focused, dependency-free tests for PDF extraction.

Run directly from the project root with::

    python tests/test_extractor_unit.py

The production extractor imports PyMuPDF, Pillow, and pytesseract lazily.  The
tests replace those modules with tiny fakes, so no PDF/OCR installation is
needed to verify ordering and failure behavior.
"""

import sys
import types
import unittest
from unittest.mock import patch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractor import _extract_docx, _extract_pdf


class _Pixmap:
    def __init__(self, text):
        self.text = text

    def tobytes(self, _format):
        return self.text.encode("utf-8")


class _Page:
    def __init__(self, text, ocr_text=None):
        self.text = text
        self.ocr_text = ocr_text

    def get_text(self, **kwargs):
        return self.text

    def get_pixmap(self, dpi):
        self.seen_dpi = dpi
        return _Pixmap(self.ocr_text or "")


class _Document:
    def __init__(self, pages):
        self.pages = pages
        self.closed = False

    def __iter__(self):
        return iter(self.pages)

    def close(self):
        self.closed = True


def _fake_modules(document, ocr=None):
    """Build fake optional modules consumed by ``_ocr_pdf_page``."""
    fitz = types.ModuleType("fitz")
    fitz.open = lambda _path: document

    pytesseract = types.ModuleType("pytesseract")
    if ocr is None:
        pytesseract.image_to_string = lambda _image: ""
    else:
        pytesseract.image_to_string = ocr

    image_module = types.ModuleType("PIL.Image")
    image_module.open = lambda stream: stream.read().decode("utf-8")
    pil = types.ModuleType("PIL")
    pil.Image = image_module
    return {"fitz": fitz, "pytesseract": pytesseract, "PIL": pil, "PIL.Image": image_module}


class ExtractPdfTests(unittest.TestCase):
    def test_docx_heading_styles_are_preserved_for_chunk_metadata(self):
        class Style:
            def __init__(self, name):
                self.name = name

        class Paragraph:
            def __init__(self, text, style):
                self.text = text
                self.style = Style(style)

        document = types.SimpleNamespace(paragraphs=[
            Paragraph("Overview", "Heading 2"),
            Paragraph("Important details", "Normal"),
        ])
        docx = types.ModuleType("docx")
        docx.Document = lambda _path: document
        with patch.dict(sys.modules, {"docx": docx}):
            result = _extract_docx("report.docx")

        self.assertEqual(result, "## Overview\n\nImportant details")

    def test_ocr_pages_stay_in_original_page_order(self):
        document = _Document(
            [
                _Page("native page one"),
                _Page("", "ocr page two"),
                _Page("native page three"),
                _Page("", "ocr page four"),
            ]
        )

        def image_to_string(_image):
            return _image

        with patch.dict(sys.modules, _fake_modules(document, image_to_string)):
            result = _extract_pdf("mixed.pdf")

        self.assertEqual(
            result,
            "native page one\focr page two\fnative page three\focr page four",
        )
        self.assertTrue(document.closed)

    def test_ocr_failure_keeps_native_pages_and_closes_document(self):
        document = _Document([_Page("native page one"), _Page(""), _Page("native page three")])

        def image_to_string(_image):
            raise RuntimeError("tesseract unavailable")

        with patch.dict(sys.modules, _fake_modules(document, image_to_string)):
            result = _extract_pdf("ocr-failure.pdf")

        self.assertEqual(result, "native page one\f\fnative page three")
        self.assertTrue(document.closed)


if __name__ == "__main__":
    unittest.main()
