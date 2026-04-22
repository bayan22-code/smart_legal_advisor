"""
PDF loader for Arabic legal documents.
Extracts and normalizes Arabic text from PDF files.
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional
import arabic_reshaper
from bidi.algorithm import get_display

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def normalize_arabic_text(text: str, apply_reversal_fix: bool = False) -> str:
    """
    Normalize Arabic text for consistency.
    - Normalize Alef variants
    - Remove tatweel (kashida)
    - Normalize whitespace
    - Remove zero-width characters
    """
    if not text or not text.strip():
        return ""

    # Normalize compatibility forms (including Arabic presentation forms)
    text = unicodedata.normalize("NFKC", text)

    # Attempt recovery for visually broken Arabic fragments only when requested.
    if apply_reversal_fix and re.search(r"[\uFB50-\uFDFF\uFE70-\uFEFC]", text):
        text = get_display(arabic_reshaper.reshape(text))

    # Normalize Alef variants to standard Alef (ا)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    # Remove tatweel (kashida) - U+0640
    text = text.replace("\u0640", "")

    # Remove zero-width characters
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\ufeff", "")  # BOM

    # Normalize multiple whitespace to single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    return text.strip()


def load_pdf_with_pdfplumber(pdf_path: Path) -> str:
    """Extract text using pdfplumber (better for complex layouts)."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
    return "\n".join(full_text)


def load_pdf_with_pypdf(pdf_path: Path) -> str:
    """Extract text using pypdf (fallback)."""
    reader = PdfReader(pdf_path)
    full_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)
    return "\n".join(full_text)


def load_pdf(
    pdf_path: str | Path,
    normalize: bool = True,
    apply_reversal_fix: bool = False,
) -> str:
    """
    Load PDF and extract all text.
    Uses pdfplumber if available, otherwise pypdf.

    Args:
        pdf_path: Path to PDF file
        normalize: Whether to normalize Arabic text

    Returns:
        Extracted and optionally normalized text
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if PDFPLUMBER_AVAILABLE:
        raw_text = load_pdf_with_pdfplumber(path)
    elif PYPDF_AVAILABLE:
        raw_text = load_pdf_with_pypdf(path)
    else:
        raise ImportError("Install pdfplumber or pypdf: pip install pdfplumber pypdf")

    if normalize:
        raw_text = normalize_arabic_text(raw_text, apply_reversal_fix=apply_reversal_fix)

    return raw_text
