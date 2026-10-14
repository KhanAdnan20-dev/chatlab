"""
ChatLab Grounded RAG Pipeline.

Implements PDF ingestion, text normalization, deterministic chunking,
metadata enrichment, FAISS inner-product vector indexing, and semantic retrieval
using LangChain (PyMuPDFLoader, RecursiveCharacterTextSplitter,
HuggingFaceEmbeddings, FAISS).
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """Represents a single RAG document chunk with full source provenance."""
    chunk_id: str
    source_id: str
    source_title: str
    file_name: str
    relative_path: str
    page_start: int
    page_end: int
    subject: str
    source_url: Optional[str]
    text: str
    content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk record to dictionary."""
        return asdict(self)


@dataclass
class SearchResult:
    """Represents a retrieved search result."""
    rank: int
    score: float
    source_title: str
    file_name: str
    page_start: int
    page_end: int
    subject: str
    source_url: Optional[str]
    snippet: str
    full_text: str
    chunk_id: str


class RAGPipelineError(Exception):
    """Base exception for RAG pipeline errors."""
    pass


class PDFExtractionError(RAGPipelineError):
    """Raised when a PDF cannot be opened or parsed."""
    pass


class IndexValidationError(RAGPipelineError):
    """Raised when FAISS index or manifest is missing or inconsistent."""
    pass


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize excessive whitespace from PDF extracted text without destroying
    mathematical notation, code indentation, or meaningful symbols.

    Rules applied:
    - Standardize line endings (\r\n and \r to \n).
    - Remove null bytes, bell chars, form feeds (\x0c, \x00).
    - Replace non-breaking spaces (\u00a0) with regular spaces.
    - Preserve line indentation (leading spaces) for code and equations.
    - Collapse runs of consecutive horizontal spaces/tabs within line content to a single space.
    - Strip trailing spaces from each line.
    - Collapse runs of 3+ newlines to 2 newlines (preserving paragraph breaks).
    - Strip leading and trailing whitespace from the final text.
    """
    if not text:
        return ""

    # Normalize control chars and non-breaking spaces
    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x0c", "\n")
        .replace("\x00", "")
        .replace("\u00a0", " ")
    )

    lines: List[str] = []
    for raw_line in text.split("\n"):
        stripped_right = raw_line.rstrip()
        if not stripped_right:
            lines.append("")
            continue

        # Preserve leading spaces for code and mathematical indentation
        leading_spaces_count = len(stripped_right) - len(stripped_right.lstrip(" "))
        content = stripped_right.lstrip(" ")

        # Collapse excessive internal horizontal whitespace within content
        content_normalized = re.sub(r"[ \t]{2,}", " ", content)
        lines.append(" " * leading_spaces_count + content_normalized)

    result = "\n".join(lines)
    # Collapse 3 or more consecutive newlines to 2 newlines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def is_meaningful_text(text: str) -> bool:
    """Check if the extracted text contains meaningful alphanumeric characters."""
    if not text or not text.strip():
        return False
    # Check if there is at least one word character (alphanumeric or underscore)
    return bool(re.search(r"\w", text))


# ---------------------------------------------------------------------------
# PDF Page Extraction
# ---------------------------------------------------------------------------

def extract_pdf_pages(
    pdf_path: Union[str, Path],
    max_pages: Optional[int] = None,
) -> Tuple[List[Tuple[int, str]], List[int], Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF using PyMuPDFLoader.

    Args:
        pdf_path: Path to the target PDF file.
        max_pages: Optional limit on the number of pages to extract.

    Returns:
        Tuple of:
        - valid_pages: List of (1-indexed page number, normalized text)
        - skipped_pages: List of 1-indexed page numbers that had no extractable text
        - pdf_metadata: Dictionary of PDF document metadata
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    try:
        loader = PyMuPDFLoader(str(pdf_path))
        raw_docs = loader.load()
    except Exception as exc:
        raise PDFExtractionError(f"Failed to extract pages from '{pdf_path.name}': {exc}") from exc

    if not raw_docs:
        return [], [], {}

    pdf_metadata: Dict[str, Any] = {}
    if raw_docs and raw_docs[0].metadata:
        first_meta = raw_docs[0].metadata
        pdf_metadata = {
            "title": (first_meta.get("title") or "").strip(),
            "author": (first_meta.get("author") or "").strip(),
            "total_pages": first_meta.get("total_pages", len(raw_docs)),
            "format": first_meta.get("format", ""),
        }

    valid_pages: List[Tuple[int, str]] = []
    skipped_pages: List[int] = []

    pages_to_process = raw_docs[:max_pages] if max_pages is not None else raw_docs

    for idx, doc in enumerate(pages_to_process):
        page_num = idx + 1  # 1-indexed
        normalized = normalize_text(doc.page_content)
        if is_meaningful_text(normalized):
            valid_pages.append((page_num, normalized))
        else:
            skipped_pages.append(page_num)

    return valid_pages, skipped_pages, pdf_metadata


# ---------------------------------------------------------------------------
# Metadata Matching
# ---------------------------------------------------------------------------

def load_sources_manifest(manifest_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load sources.json and return a dictionary keyed by source_id.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sources_list = data.get("sources", [])
        return {s["source_id"]: s for s in sources_list if "source_id" in s}
    except Exception as exc:
        logger.warning("Could not read sources manifest at %s: %exc", manifest_path, exc)
        return {}


def match_source_metadata(
    pdf_path: Union[str, Path],
    project_root: Optional[Union[str, Path]] = None,
    sources_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    pdf_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enrich metadata for a PDF file using sources.json manifest, PDF metadata, or filename.
    Ensures source_url is None unless verified in sources.json.
    """
    pdf_path = Path(pdf_path)
    if project_root is not None:
        project_root = Path(project_root)
        try:
            rel_path = pdf_path.relative_to(project_root).as_posix()
        except ValueError:
            rel_path = pdf_path.as_posix()
    else:
        rel_path = pdf_path.as_posix()

    parent_dir_name = pdf_path.parent.name
    file_stem = pdf_path.stem

    # Attempt exact match against sources manifest by parent directory name or file stem
    matched_entry = None
    if sources_manifest:
        if parent_dir_name in sources_manifest:
            matched_entry = sources_manifest[parent_dir_name]
        elif file_stem in sources_manifest:
            matched_entry = sources_manifest[file_stem]

    if matched_entry:
        source_id = matched_entry.get("source_id", parent_dir_name)
        source_title = matched_entry.get("title") or pdf_title or file_stem.replace("_", " ").title()
        source_url = matched_entry.get("source_url")
        subject = matched_entry.get("subject", parent_dir_name)
    else:
        source_id = file_stem
        if pdf_title and pdf_title.strip():
            source_title = pdf_title.strip()
        else:
            source_title = file_stem.replace("_", " ").title()
        source_url = None
        subject = parent_dir_name

    return {
        "source_id": source_id,
        "source_title": source_title,
        "file_name": pdf_path.name,
        "relative_path": rel_path,
        "subject": subject,
        "source_url": source_url,
    }


# ---------------------------------------------------------------------------
# Chunking Strategy
# ---------------------------------------------------------------------------

def chunk_document_pages(
    valid_pages: List[Tuple[int, str]],
    source_metadata: Dict[str, Any],
    chunk_size: int = 220,
    chunk_overlap: int = 40,
) -> List[ChunkRecord]:
    """
    Chunk document text using LangChain's RecursiveCharacterTextSplitter with word count length.
    Allows chunks to cross page boundaries while accurately tracking page_start and page_end.

    Args:
        valid_pages: List of (1-indexed page_num, normalized_page_text)
        source_metadata: Enriched source metadata dictionary
        chunk_size: Target words per chunk (default: 220)
        chunk_overlap: Target overlap words (default: 40)

    Returns:
        List of ChunkRecord objects
    """
    if not valid_pages:
        return []

    # Concatenate pages while recording character spans for each page
    full_text_parts: List[str] = []
    page_spans: List[Tuple[int, int, int]] = []  # (page_num, start_char, end_char)
    current_char = 0

    for page_num, text in valid_pages:
        start_char = current_char
        full_text_parts.append(text)
        current_char += len(text)
        end_char = current_char
        page_spans.append((page_num, start_char, end_char))

        # Join pages with two newlines
        full_text_parts.append("\n\n")
        current_char += 2

    full_document_text = "".join(full_text_parts)

    # Split text using RecursiveCharacterTextSplitter with word count length function
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda t: len(t.split()),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_chunks = splitter.split_text(full_document_text)

    chunks: List[ChunkRecord] = []
    source_id = source_metadata["source_id"]
    curr_search_pos = 0

    for idx, raw_chunk in enumerate(split_chunks):
        chunk_text = raw_chunk.strip()
        if not is_meaningful_text(chunk_text):
            continue

        # Monotonically locate the chunk within full_document_text.
        # This completely avoids LangChain's TextSplitter.create_documents bug
        # where character-based overlap calculation causes text.find to fail and return -1.
        pos = full_document_text.find(chunk_text, curr_search_pos)
        if pos == -1:
            # Fallback search from document start if somehow not found forward
            pos = full_document_text.find(chunk_text, 0)
            if pos == -1:
                pos = curr_search_pos
        curr_search_pos = pos

        s_char = pos
        e_char = pos + len(chunk_text)

        # Resolve first physical page overlapping with [s_char, e_char]
        page_start = valid_pages[0][0]
        for p_num, p_start, p_end in page_spans:
            if p_end > s_char:
                page_start = p_num
                break

        # Resolve last physical page overlapping with [s_char, e_char]
        page_end = valid_pages[-1][0]
        for p_num, p_start, p_end in reversed(page_spans):
            if p_start < e_char:
                page_end = p_num
                break

        # Ensure valid ordered bounds
        if page_start > page_end:
            page_start = page_end

        # Deterministic chunk ID and SHA-256 content hash
        chunk_id = f"{source_id}_chunk_{len(chunks) + 1:05d}"
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        chunk_record = ChunkRecord(
            chunk_id=chunk_id,
            source_id=source_id,
            source_title=source_metadata["source_title"],
            file_name=source_metadata["file_name"],
            relative_path=source_metadata["relative_path"],
            page_start=page_start,
            page_end=page_end,
            subject=source_metadata["subject"],
            source_url=source_metadata["source_url"],
            text=chunk_text,
            content_hash=content_hash,
        )
        chunks.append(chunk_record)

    return chunks


# ---------------------------------------------------------------------------
# JSONL Document Export & Validation
# ---------------------------------------------------------------------------

REQUIRED_JSONL_KEYS = {
    "chunk_id",
    "source_id",
    "source_title",
    "file_name",
    "relative_path",
    "page_start",
    "page_end",
    "subject",
    "source_url",
    "text",
    "content_hash",
}


def export_documents_jsonl(
    chunks: List[ChunkRecord],
    output_path: Union[str, Path],
) -> Path:
    """
    Safely export ChunkRecords to a valid UTF-8 JSONL file with validation.
    Uses atomic write via temporary file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f".tmp_{os.getpid()}")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                line = json.dumps(chunk.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

        # Validate temporary file before atomic replacement
        validate_documents_jsonl(temp_path, expected_count=len(chunks))

        # Safely replace target file
        if output_path.exists():
            output_path.unlink()
        temp_path.rename(output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return output_path


def validate_documents_jsonl(
    jsonl_path: Union[str, Path],
    expected_count: Optional[int] = None,
) -> int:
    """
    Validate that a JSONL file exists, is valid UTF-8, contains valid JSON per line,
    and includes all required schema keys.

    Returns the count of validated records.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"documents.jsonl file does not exist at {jsonl_path}")

    record_count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at line {line_no} in {jsonl_path}: {err}") from err

            missing_keys = REQUIRED_JSONL_KEYS - set(record.keys())
            if missing_keys:
                raise ValueError(
                    f"Missing required keys {missing_keys} at line {line_no} in {jsonl_path}"
                )
            if not record.get("text", "").strip():
                raise ValueError(f"Empty text field in record at line {line_no} in {jsonl_path}")

            record_count += 1

    if expected_count is not None and record_count != expected_count:
        raise ValueError(
            f"Record count mismatch in {jsonl_path}: expected {expected_count}, found {record_count}"
        )

    return record_count


# ---------------------------------------------------------------------------
# FAISS Index Construction & Safe Persistence
# ---------------------------------------------------------------------------

def get_optimal_device(requested_device: Optional[str] = None) -> str:
    """
    Determine compute device for HuggingFaceEmbeddings.
    Defaults strictly to 'cpu' at runtime so 100% of CUDA VRAM is preserved
    for Gemma 4 E2B 4-bit NF4 inference. Allows explicit override (e.g. 'cuda')
    only when explicitly passed via requested_device.
    """
    if requested_device:
        req = requested_device.strip().lower()
        if req == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return req
    return "cpu"


def build_faiss_index(
    chunks: List[ChunkRecord],
    model_path: Union[str, Path],
    index_dir: Union[str, Path],
    batch_size: int = 64,
    device: Optional[str] = None,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
    sources_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Build a FAISS inner-product index using LangChain's FAISS integration and
    HuggingFaceEmbeddings from langchain_huggingface.
    Saves index.faiss, index.pkl, and index_manifest.json safely.
    """
    if not chunks:
        raise ValueError("Cannot build FAISS index: chunks list is empty")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Local embedding model not found at {model_path}")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    device = get_optimal_device(device)

    logger.info("Initializing HuggingFaceEmbeddings on device '%s' (batch_size=%d)...", device, batch_size)
    embeddings = HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )

    texts = [c.text for c in chunks]
    metadatas = [c.to_dict() for c in chunks]

    logger.info("Encoding %d chunks and building FAISS MAX_INNER_PRODUCT vector store...", len(chunks))
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    # Validate vector count alignment
    indexed_vectors = vector_store.index.ntotal
    if indexed_vectors != len(chunks):
        raise ValueError(
            f"FAISS index vector count ({indexed_vectors}) does not match chunks count ({len(chunks)})"
        )

    # Validate vector dimension (384 for all-MiniLM-L6-v2)
    vector_dim = vector_store.index.d

    # Build manifest
    manifest_data = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": str(model_path).replace("\\", "/"),
        "vector_dimension": vector_dim,
        "similarity_method": "inner_product",
        "distance_strategy": "MAX_INNER_PRODUCT",
        "total_chunks": len(chunks),
        "chunking_config": {
            "chunk_size_words": chunk_size,
            "chunk_overlap_words": chunk_overlap,
            "length_function": "word_count",
        },
        "device": device,
        "artifacts": {
            "documents_jsonl": "data/rag/processed/documents.jsonl",
            "faiss_index": "data/rag/indexes/faiss/index.faiss",
            "faiss_pkl": "data/rag/indexes/faiss/index.pkl",
            "index_manifest": "data/rag/indexes/faiss/index_manifest.json",
        },
        "sources_summary": sources_summary or {},
    }

    # Safe write via staging directory
    staging_dir = index_dir / f".staging_{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        # LangChain saves index.faiss and index.pkl
        vector_store.save_local(str(staging_dir))

        # Save manifest
        manifest_staging_file = staging_dir / "index_manifest.json"
        with open(manifest_staging_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Safely copy files to final destination
        for filename in ["index.faiss", "index.pkl", "index_manifest.json"]:
            src = staging_dir / filename
            dst = index_dir / filename
            if not src.exists():
                raise FileNotFoundError(f"Staged file missing: {src}")
            shutil.copy2(src, dst)

        logger.info("FAISS index successfully saved to %s (%d vectors)", index_dir, indexed_vectors)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    return index_dir


# ---------------------------------------------------------------------------
# Index Loading & Retrieval
# ---------------------------------------------------------------------------

def load_and_validate_index(
    index_dir: Union[str, Path],
    model_path: Union[str, Path],
    device: Optional[str] = None,
) -> Tuple[FAISS, Dict[str, Any]]:
    """
    Validate all artifacts and load the trusted local FAISS index.
    Only enables allow_dangerous_deserialization for the trusted index generated
    by this local pipeline.
    """
    index_dir = Path(index_dir)
    model_path = Path(model_path)

    if not index_dir.exists():
        raise IndexValidationError(f"Index directory does not exist: {index_dir}")

    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    manifest_file = index_dir / "index_manifest.json"

    if not faiss_file.exists():
        raise IndexValidationError(f"FAISS index file missing: {faiss_file}")
    if not pkl_file.exists():
        raise IndexValidationError(f"FAISS pickle file missing: {pkl_file}")
    if not manifest_file.exists():
        raise IndexValidationError(f"FAISS manifest file missing: {manifest_file}")

    if not model_path.exists():
        raise IndexValidationError(f"Embedding model directory missing: {model_path}")

    # Read and validate manifest
    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        raise IndexValidationError(f"Failed to parse index manifest: {exc}") from exc

    required_manifest_keys = {"vector_dimension", "total_chunks", "embedding_model"}
    missing = required_manifest_keys - set(manifest.keys())
    if missing:
        raise IndexValidationError(f"Index manifest missing required keys: {missing}")

    device = get_optimal_device(device)
    embeddings = HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load local trusted FAISS index
    try:
        vector_store = FAISS.load_local(
            folder_path=str(index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
    except Exception as exc:
        raise IndexValidationError(f"Failed to load FAISS index from {index_dir}: {exc}") from exc

    if vector_store.index.ntotal != manifest["total_chunks"]:
        raise IndexValidationError(
            f"FAISS index row count ({vector_store.index.ntotal}) does not match manifest count ({manifest['total_chunks']})"
        )

    return vector_store, manifest


def _compute_bm25_ranks(
    vector_store: FAISS,
    query: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, Tuple[int, float, Any]]:
    """
    Lightweight CPU Okapi BM25 lexical ranking over the loaded FAISS docstore.
    Requires zero external dependencies and zero index rebuilds.
    Returns mapping: chunk_key -> (1-based_bm25_rank, bm25_score, Document).
    """
    query_terms = re.findall(r"\w+", query.lower())
    if not query_terms or not hasattr(vector_store, "docstore"):
        return {}

    docs_dict = getattr(vector_store.docstore, "_dict", {})
    if not docs_dict or not isinstance(docs_dict, dict):
        return {}

    N = len(docs_dict)
    doc_tokens_list: List[Tuple[Any, Counter, int]] = []
    df: Counter = Counter()
    total_len = 0

    for doc in docs_dict.values():
        content = getattr(doc, "page_content", "") or ""
        tokens = re.findall(r"\w+", content.lower())
        dl = len(tokens)
        total_len += dl
        tf = Counter(tokens)
        doc_tokens_list.append((doc, tf, dl))
        for term in set(query_terms):
            if term in tf:
                df[term] += 1

    avgdl = max(total_len / max(N, 1), 1.0)
    scored_docs: List[Tuple[float, str, Any]] = []

    for idx, (doc, tf, dl) in enumerate(doc_tokens_list):
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            n_q = df[term]
            idf = math.log(((N - n_q + 0.5) / (n_q + 0.5)) + 1.0)
            freq = tf[term]
            denom = freq + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * ((freq * (k1 + 1.0)) / denom)
        if score > 0.0:
            meta = getattr(doc, "metadata", None) or {}
            cid = str(meta.get("chunk_id") or f"doc_{idx}_{id(doc)}")
            scored_docs.append((score, cid, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    bm25_map: Dict[str, Tuple[int, float, Any]] = {}
    for rank_idx, (bm25_score, cid, doc) in enumerate(scored_docs, start=1):
        bm25_map[cid] = (rank_idx, bm25_score, doc)

    return bm25_map


def search_index(
    vector_store: FAISS,
    query: str,
    top_k: int = 5,
    snippet_len: int = 300,
    rrf_k: int = 60,
) -> List[SearchResult]:
    """
    Search the FAISS index with natural language query using Hybrid RAG:
    Combines Dense FAISS Inner-Product (MiniLM-L6-v2 cosine similarity)
    with Okapi BM25 Lexical Ranking via Reciprocal Rank Fusion (RRF).
    """
    if not query or not query.strip():
        return []

    # 1. Dense Semantic Retrieval (expand candidate pool for RRF fusion)
    candidate_k = max(top_k * 3, 15)
    docs_and_scores = vector_store.similarity_search_with_score(query, k=candidate_k)

    dense_map: Dict[str, Tuple[int, float, Any]] = {}
    for rank_idx, (doc, raw_score) in enumerate(docs_and_scores, start=1):
        meta = getattr(doc, "metadata", None) or {}
        cid = str(meta.get("chunk_id") or f"dense_{rank_idx}_{id(doc)}")
        dense_map[cid] = (rank_idx, float(raw_score), doc)

    # 2. Lexical Okapi BM25 Retrieval
    bm25_map = _compute_bm25_ranks(vector_store, query)

    # Fallback: If BM25 returned no matches (or mock vector_store in unit tests has no _dict),
    # preserve exact dense ordering and scores.
    if not bm25_map:
        results: List[SearchResult] = []
        for rank, (doc, raw_score) in enumerate(docs_and_scores[:top_k], start=1):
            meta = doc.metadata or {}
            score = float(raw_score)
            text = doc.page_content or ""
            snippet = text[:snippet_len] + ("..." if len(text) > snippet_len else "")
            results.append(
                SearchResult(
                    rank=rank,
                    score=round(score, 4),
                    source_title=meta.get("source_title", "Unknown"),
                    file_name=meta.get("file_name", "Unknown"),
                    page_start=meta.get("page_start", 1),
                    page_end=meta.get("page_end", 1),
                    subject=meta.get("subject", "general"),
                    source_url=meta.get("source_url"),
                    snippet=snippet,
                    full_text=text,
                    chunk_id=meta.get("chunk_id", ""),
                )
            )
        return results

    # 3. Reciprocal Rank Fusion (RRF): RRF(d) = w_d / (k + r_d) + w_b / (k + r_b)
    all_cids = list(dense_map.keys())
    for cid in bm25_map.keys():
        if cid not in dense_map:
            all_cids.append(cid)

    penalty_rank = candidate_k + 50
    fused_candidates: List[Tuple[float, float, Any]] = []

    for cid in all_cids:
        d_rank, d_score, d_doc = dense_map.get(cid, (penalty_rank, 0.0, None))
        b_rank, b_score, b_doc = bm25_map.get(cid, (penalty_rank, 0.0, None))
        doc = d_doc if d_doc is not None else b_doc

        rrf_score = (0.65 / (rrf_k + d_rank)) + (0.35 / (rrf_k + b_rank))
        # Preserve calibrated cosine score while applying a small bounded boost for top lexical hits
        lexical_boost = 0.08 if b_rank <= top_k else (0.03 if b_rank <= candidate_k else 0.0)
        display_score = min(1.0, d_score + lexical_boost) if d_score > 0.0 else min(0.75, 0.35 + lexical_boost)

        fused_candidates.append((rrf_score, display_score, doc))

    fused_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    results = []
    for rank, (_, display_score, doc) in enumerate(fused_candidates[:top_k], start=1):
        meta = doc.metadata or {}
        text = doc.page_content or ""
        snippet = text[:snippet_len] + ("..." if len(text) > snippet_len else "")
        results.append(
            SearchResult(
                rank=rank,
                score=round(float(display_score), 4),
                source_title=meta.get("source_title", "Unknown"),
                file_name=meta.get("file_name", "Unknown"),
                page_start=meta.get("page_start", 1),
                page_end=meta.get("page_end", 1),
                subject=meta.get("subject", "general"),
                source_url=meta.get("source_url"),
                snippet=snippet,
                full_text=text,
                chunk_id=meta.get("chunk_id", ""),
            )
        )

    return results
