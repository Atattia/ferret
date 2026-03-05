from pathlib import Path


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF using pymupdf, falling back to OCR for scanned pages."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        pages_text = []
        ocr_needed = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages_text.append(text)
            else:
                ocr_needed.append(i)

        # OCR scanned pages
        if ocr_needed:
            try:
                import pytesseract
                from PIL import Image
                import io
                for i in ocr_needed:
                    page = doc[i]
                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if ocr_text:
                        pages_text.append(ocr_text)
            except Exception as e:
                print(f"[extractor] OCR fallback failed for {path}: {e}")

        doc.close()
        return "\n".join(pages_text)
    except Exception as e:
        print(f"[extractor] PDF extraction failed for {path}: {e}")
        return ""


def _extract_docx(path: Path) -> str:
    """Extract text from .docx files."""
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[extractor] DOCX extraction failed for {path}: {e}")
        return ""


def _extract_plain(path: Path) -> str:
    """Extract text from plain text files."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[extractor] Plain text read failed for {path}: {e}")
            return ""
    return ""


def extract(path: str | Path) -> str:
    """Extract text from a file based on its extension. Returns empty string on failure."""
    path = Path(path)
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
