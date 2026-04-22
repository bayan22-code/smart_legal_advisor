"""
Shared Arabic text normalization for ingestion and retrieval.
"""

from __future__ import annotations

import re
import unicodedata

import arabic_reshaper
from bidi.algorithm import get_display


def clean_text(text: str, apply_reversal_fix: bool = False) -> str:
    """
    Normalize Arabic text into a stable logical form for indexing and querying.

    apply_reversal_fix should only be enabled for known reversed sources.
    """
    if text is None:
        return ""

    cleaned = str(text)
    cleaned = cleaned.replace("\ufeff", "").replace("\u200f", "").replace("\u200e", "")
    cleaned = cleaned.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")

    # Convert Arabic presentation forms to base alphabetic forms.
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # Normalize whitespace while preserving line boundaries.
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    if apply_reversal_fix:
        cleaned = get_display(arabic_reshaper.reshape(cleaned))

    return cleaned
