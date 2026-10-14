"""
ChatLab Grounded RAG Pipeline Package.
Provides PDF ingestion, chunking, metadata enrichment, FAISS indexing, and retrieval.
"""

from backend.app.rag.pipeline import (
    ChunkRecord,
    SearchResult,
    normalize_text,
    extract_pdf_pages,
    match_source_metadata,
    chunk_document_pages,
    export_documents_jsonl,
    build_faiss_index,
    load_and_validate_index,
    search_index,
)

__all__ = [
    "ChunkRecord",
    "SearchResult",
    "normalize_text",
    "extract_pdf_pages",
    "match_source_metadata",
    "chunk_document_pages",
    "export_documents_jsonl",
    "build_faiss_index",
    "load_and_validate_index",
    "search_index",
]
