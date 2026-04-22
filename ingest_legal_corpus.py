"""
Build a high-quality Arabic legal corpus for RAG from multiple Syrian law PDFs.

Features:
- PDF extraction per page (pdfplumber with pypdf fallback)
- Recurring header/footer removal (including UNESCO DISCLAIMER)
- Page number stripping (Arabic and Western digits)
- Article-level chunking with regex on "المادة <number>"
- Arabic integrity processing with arabic-reshaper + python-bidi
- Metadata enrichment (law source + article number)
- ChromaDB ingestion with multilingual legal-friendly embeddings
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arabic_reshaper
import chromadb
from bidi.algorithm import get_display
from sentence_transformers import SentenceTransformer
from text_normalization import clean_text

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


ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ARABIC_TO_WESTERN = str.maketrans(ARABIC_DIGITS, "0123456789")
ARTICLE_REGEX = re.compile(
    r"(?m)^\s*(المادة\s*(?:رقم\s*)?([0-9\u0660-\u0669]+)[^\n]*)"
)
ARTICLE_SPLIT_PATTERN = r"(?=المادة\s+\d+)"
PAGE_NUMBER_REGEX = re.compile(
    r"^\s*(?:صفحة\s*)?[0-9\u0660-\u0669]{1,4}(?:\s*/\s*[0-9\u0660-\u0669]{1,4})?\s*$"
)
UNESCO_REGEX = re.compile(r"UNESCO\s+DISCLAIMER", re.IGNORECASE)


@dataclass(frozen=True)
class LawDoc:
    file_name: str
    source: str
    law_name: str


LAW_DOCS = [
    LawDoc("sy_penalcode_49_arorof.pdf", "Penal Law", "قانون العقوبات"),
    LawDoc("personal_statlaw.pdf", "Personal Status Law", "قانون الأحوال الشخصية"),
    LawDoc("civil_law.pdf", "Civil Law", "القانون المدني"),
]


def normalize_text_basic(text: str) -> str:
    return clean_text(text, apply_reversal_fix=False)


def repair_broken_arabic(text: str, apply_reversal_fix: bool) -> str:
    """
    Repair Arabic presentation forms and visual-order artifacts from PDFs.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    has_presentation_forms = bool(re.search(r"[\uFB50-\uFDFF\uFE70-\uFEFC]", normalized))
    if apply_reversal_fix and has_presentation_forms:
        # Convert visual-order fragments to better logical order.
        logical_guess = get_display(normalized)
        normalized = logical_guess
    return normalized


def normalize_digits_for_split(text: str) -> str:
    """Convert Arabic-Indic digits to western digits so \\d+ split works."""
    return text.translate(ARABIC_TO_WESTERN)


def to_western_number(raw_number: str) -> int | None:
    translated = raw_number.translate(ARABIC_TO_WESTERN)
    digits = re.sub(r"\D+", "", translated)
    return int(digits) if digits else None


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    if PDFPLUMBER_AVAILABLE:
        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return pages

    if PYPDF_AVAILABLE:
        reader = PdfReader(str(pdf_path))
        return [(page.extract_text() or "") for page in reader.pages]

    raise ImportError("Install pdfplumber or pypdf: pip install pdfplumber pypdf")


def detect_recurring_headers_footers(pages: list[str]) -> set[str]:
    if not pages:
        return set()

    boundary_counts: dict[str, int] = {}
    for page in pages:
        raw_lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not raw_lines:
            continue

        boundary = raw_lines[:3] + raw_lines[-3:]
        for line in boundary:
            normalized = normalize_text_basic(line)
            if not normalized:
                continue
            boundary_counts[normalized] = boundary_counts.get(normalized, 0) + 1

    min_repeats = max(3, math.ceil(len(pages) * 0.25))
    recurring = {
        line
        for line, count in boundary_counts.items()
        if count >= min_repeats
        and len(line) <= 120
        and "المادة" not in line
        and not PAGE_NUMBER_REGEX.match(line)
    }
    recurring.add("UNESCO DISCLAIMER")
    return recurring


def clean_page_text(page_text: str, recurring_lines: set[str]) -> str:
    cleaned_lines: list[str] = []
    for raw_line in page_text.splitlines():
        line = normalize_text_basic(raw_line)
        if not line:
            continue
        if UNESCO_REGEX.search(line):
            continue
        if PAGE_NUMBER_REGEX.match(line):
            continue
        if line in recurring_lines:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def process_arabic_integrity(text: str, apply_reversal_fix: bool = False) -> tuple[str, str]:
    repaired = repair_broken_arabic(text, apply_reversal_fix=apply_reversal_fix)
    logical = normalize_text_basic(repaired)
    visual = logical
    if apply_reversal_fix:
        shaped = arabic_reshaper.reshape(logical)
        visual = get_display(shaped)
    return logical, visual


def law_name_from_filename(file_name: str) -> str:
    lowered = file_name.lower()
    if "penal" in lowered or "عقوبات" in lowered:
        return "قانون العقوبات"
    if "personal" in lowered or "احوال" in lowered:
        return "قانون الأحوال الشخصية"
    if "civil" in lowered or "مدني" in lowered:
        return "القانون المدني"
    return "القانون السوري"


def chunk_by_article(
    text: str,
    source: str,
    law_name: str,
    apply_reversal_fix: bool = False,
) -> list[dict[str, Any]]:
    normalized_for_split = normalize_digits_for_split(text)
    parts = [part.strip() for part in re.split(ARTICLE_SPLIT_PATTERN, normalized_for_split) if part and part.strip()]
    chunks: list[dict[str, Any]] = []

    if not parts:
        logical, visual = process_arabic_integrity(text, apply_reversal_fix=apply_reversal_fix)
        return [
            {
                "text": logical,
                "metadata": {
                    "source": source,
                    "law_name": law_name,
                    "article_number": 0,
                    "article_title": "NO_ARTICLE_MARKER",
                },
                "text_visual": visual,
            }
        ]

    for idx, part in enumerate(parts, start=1):
        if not part.startswith("المادة"):
            continue
        title_match = ARTICLE_REGEX.search(part)
        article_title = title_match.group(1).strip() if title_match else part.split("\n", 1)[0].strip()
        article_number = to_western_number(title_match.group(2)) if title_match else None
        article_number = article_number or idx
        logical, visual = process_arabic_integrity(part, apply_reversal_fix=apply_reversal_fix)
        if not logical:
            continue
        chunks.append(
            {
                "text": logical,
                "metadata": {
                    "source": source,
                    "law_name": law_name,
                    "article_number": article_number,
                    "article_title": article_title,
                    "part_index": 0,
                },
                "text_visual": visual,
            }
        )
    return chunks


def embed_chunks(texts: list[str], model_name: str) -> list[list[float]]:
    model = SentenceTransformer(model_name)
    prefixed = [f"passage: {text}" for text in texts]
    vectors = model.encode(prefixed, batch_size=16, show_progress_bar=True)
    return vectors.tolist()


def ingest_into_chroma(
    chunks: list[dict[str, Any]],
    db_dir: Path,
    collection_name: str,
    embedding_model: str,
) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
    )

    documents = [chunk["text"] for chunk in chunks]
    embeddings = embed_chunks(documents, model_name=embedding_model)
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [
        f"{meta['source'].lower().replace(' ', '_')}_article_{meta['article_number']}_{i}"
        for i, meta in enumerate(metadatas, start=1)
    ]

    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def build_corpus(data_dir: Path) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []

    for law in LAW_DOCS:
        pdf_path = data_dir / law.file_name
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")

        pages = extract_pdf_pages(pdf_path)
        recurring = detect_recurring_headers_footers(pages)
        cleaned_pages = [clean_page_text(page, recurring) for page in pages]
        merged_text = normalize_text_basic("\n\n".join(page for page in cleaned_pages if page))
        detected_law_name = law_name_from_filename(law.file_name)
        chunks = chunk_by_article(
            merged_text,
            source=law.source,
            law_name=detected_law_name,
            apply_reversal_fix=False,
        )
        all_chunks.extend(chunks)
        print(f"{law.source}: {len(chunks)} article chunks")

    return all_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Syrian law PDFs into ChromaDB.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing the three law PDFs.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "vector_db",
        help="Chroma persistent storage directory.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="syrian_laws_rag",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Multilingual embedding model suited for Arabic legal text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = build_corpus(args.data_dir)
    ingest_into_chroma(
        chunks=chunks,
        db_dir=args.db_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    print(f"Ingestion complete. Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
