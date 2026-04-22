"""
Smart legal article chunker for Syrian Civil Law.
Splits document by article pattern - each chunk = one legal article.
"""

import re
from typing import Any

from config import ARTICLE_PATTERN

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
MAX_ARTICLE_CHARS = 2500


def _to_western_digits(value: str) -> str:
    table = str.maketrans(ARABIC_INDIC_DIGITS, "0123456789")
    return value.translate(table)


def _extract_article_number(article_header: str, fallback: int) -> int:
    normalized = _to_western_digits(article_header)
    num_match = re.search(r"\d+", normalized)
    if not num_match:
        return fallback
    try:
        return int(num_match.group())
    except ValueError:
        return fallback


def _clean_article_text(article_text: str) -> str:
    """
    Remove boilerplate/disclaimer lines while preserving legal content.
    """
    if not article_text:
        return ""
    text = article_text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Remove common non-legal headers/footers from extracted PDFs.
    noisy_patterns = [
        r"^الصفحة\s+\d+.*$",
        r"^Page\s+\d+.*$",
        r"^جميع الحقوق محفوظة.*$",
        r"^تم التحميل من.*$",
        r"^هذا النص للاطلاع.*$",
    ]
    lines = []
    for line in text.split("\n"):
        ln = line.strip()
        if not ln:
            continue
        if any(re.match(p, ln, flags=re.IGNORECASE) for p in noisy_patterns):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def _subsplit_long_article(article_text: str, max_chars: int = MAX_ARTICLE_CHARS) -> list[str]:
    """
    Keep article intact when possible; only split very long articles by paragraph breaks.
    Each sub-chunk preserves the article header.
    """
    if len(article_text) <= max_chars:
        return [article_text]

    lines = [ln for ln in article_text.split("\n") if ln.strip()]
    header = lines[0].strip() if lines else "المادة"
    body_lines = lines[1:] if len(lines) > 1 else []
    if not body_lines:
        return [article_text]

    chunks: list[str] = []
    current = f"{header}\n"
    for line in body_lines:
        candidate = f"{current}{line}\n"
        if len(candidate) > max_chars and current.strip() != header:
            chunks.append(current.strip())
            current = f"{header}\n{line}\n"
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [article_text]


def chunk_legal_articles(
    text: str,
    law_name: str = "القانون المدني السوري",
    source: str = "civil_law.pdf",
    language: str = "arabic",
) -> list[dict[str, Any]]:
    """
    Split document into legal article chunks using regex pattern.

    Args:
        text: Full document text
        law_name: Name of the law
        source: Source file name
        language: Language code

    Returns:
        List of chunks, each containing one article with metadata
    """
    # Semantic split by legal article boundaries.
    # Equivalent to re.split(r'(?=المادة\s+\d+)') with Arabic-Indic digit support.
    split_pattern = ARTICLE_PATTERN or r"(?=المادة\s+[0-9٠-٩]+)"
    article_parts = [part.strip() for part in re.split(split_pattern, text) if part and part.strip()]
    if not article_parts:
        cleaned = _clean_article_text(text)
        return [{"text": cleaned, "metadata": _make_metadata(0, law_name, source, language)}]

    chunks = []
    for i, part in enumerate(article_parts, start=1):
        if not part.startswith("المادة"):
            continue
        article_text = _clean_article_text(part)
        if not article_text:
            continue
        article_number = _extract_article_number(article_text.split("\n", 1)[0], i)
        sub_chunks = _subsplit_long_article(article_text)
        for part_idx, part_text in enumerate(sub_chunks):
            chunks.append(
                {
                    "text": part_text,
                    "metadata": _make_metadata(
                        article_number=article_number,
                        law_name=law_name,
                        source=source,
                        language=language,
                        part_index=part_idx,
                        total_parts=len(sub_chunks),
                    ),
                }
            )

    if not chunks:
        cleaned = _clean_article_text(text)
        return [{"text": cleaned, "metadata": _make_metadata(0, law_name, source, language)}]

    return chunks


def _make_metadata(
    article_number: int,
    law_name: str,
    source: str,
    language: str,
    part_index: int = 0,
    total_parts: int = 1,
) -> dict[str, Any]:
    """Build metadata dict for a chunk."""
    return {
        "article_number": article_number,
        "law": law_name,
        "source": source,
        "language": language,
        "part_index": part_index,
        "total_parts": total_parts,
    }
