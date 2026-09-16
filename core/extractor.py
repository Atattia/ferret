from pathlib import Path
from contextvars import ContextVar
import os

_warnings = ContextVar("extraction_warnings", default=())


def configure_ocr(config):
    os.environ["FERRET_OCR_LANGUAGES"] = config.get("ocr_languages", "ara+eng")
    import sys
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    local = Path(config.get("ocr_data_path", str(base / "models" / "tessdata"))).expanduser()
    if local.is_dir():
        os.environ["TESSDATA_PREFIX"] = str(local)


def extraction_warnings():
    return list(_warnings.get())


def _warn(message):
    _warnings.set((*_warnings.get(), message))


def _usable_text(text):
    if not text.strip():
        return False
    broken = text.count("\ufffd") + text.count("(cid:") * 5
    return broken / max(1, len(text)) < 0.05


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF using pymupdf, falling back to OCR for scanned pages."""
    doc = None
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        pages_text = []
        # Process each page completely before moving on to the next one.  In
        # particular, OCR text must be inserted at the scanned page's original
        # position; collecting OCR pages separately would reorder mixed PDFs.
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text(sort=True).strip()
            if _usable_text(text):
                pages_text.append(text)
                continue

            try:
                ocr_text = _ocr_pdf_page(page)
            except Exception as e:
                _warn(f"Page {page_number}: OCR failed: {e}")
                # A broken/missing OCR installation should not discard native
                # text extracted from the other pages.
                print(
                    f"[extractor] OCR fallback failed for {path} "
                    f"(page {page_number}): {e}"
                )
                pages_text.append("")
                continue
            if not ocr_text.strip():
                _warn(f"Page {page_number}: no readable text")
            pages_text.append(ocr_text or "")

        # Form feed is an internal page boundary understood by the chunker.
        # It is removed before embedding/search display, while preserving the
        # source page number for navigation metadata.
        return "\f".join(pages_text)
    except Exception as e:
        _warn(f"PDF extraction failed: {e}")
        print(f"[extractor] PDF extraction failed for {path}: {e}")
        return ""
    finally:
        if doc is not None:
            doc.close()


def _ocr_pdf_page(page) -> str:
    """Render and OCR one PDF page, returning normalized text.

    Imports remain local so OCR stays an optional dependency for users who
    only search native PDFs and plain documents.  Exceptions intentionally
    propagate to ``_extract_pdf`` where they can be reported with a page
    number while preserving the rest of the document.
    """
    import io

    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=250)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    languages = os.environ.get("FERRET_OCR_LANGUAGES", "")
    if languages:
        available = set(pytesseract.get_languages(config=""))
        missing = set(languages.split("+")) - available
        if missing:
            raise RuntimeError("Missing OCR language packs: " + ", ".join(sorted(missing)))
        return pytesseract.image_to_string(image, lang=languages).strip()
    return pytesseract.image_to_string(image).strip()


def _extract_docx(path: Path) -> str:
    """Extract text from .docx files."""
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = []
        blocks = doc.iter_inner_content() if hasattr(doc, "iter_inner_content") else doc.paragraphs
        for paragraph in blocks:
            if hasattr(paragraph, "rows"):
                for row in paragraph.rows:
                    paragraphs.append(" | ".join(cell.text for cell in row.cells))
                continue
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = getattr(getattr(paragraph, "style", None), "name", "")
            if style_name.startswith("Heading "):
                try:
                    level = min(6, max(1, int(style_name.split()[-1])))
                    text = f"{'#' * level} {text}"
                except ValueError:
                    pass
            paragraphs.append(text)
        return "\n\n".join(paragraphs)
    except Exception as e:
        _warn(f"DOCX extraction failed: {e}")
        print(f"[extractor] DOCX extraction failed for {path}: {e}")
        return ""


def _extract_plain(path: Path) -> str:
    """Extract text from plain text files."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _warn(f"Could not read file: {exc}")
        return ""
    encodings = ("utf-16",) if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "cp1256", "cp1252")
    for encoding in encodings:
        try:
            decoded = raw.decode(encoding)
            if encoding in {"cp1256", "cp1252"}:
                _warn(f"Legacy {encoding} decoding used; verify the document preview")
            return decoded
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[extractor] Plain text read failed for {path}: {e}")
            return ""
    return ""


def extract(path: str | Path) -> str:
    """Extract text from a file based on its extension. Returns empty string on failure."""
    path = Path(path)
    _warnings.set(())
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix == ".docx":
        return _extract_docx(path)
    elif suffix in (".txt", ".md"):
        return _extract_plain(path)
    else:
        # Attempt plain text read for unknown types
        return _extract_plain(path)
