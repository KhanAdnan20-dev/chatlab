"""
Unit tests for ChatLab RAG Pipeline (PDF Extraction, Chunking, Metadata, FAISS Indexing, Retrieval).
Tests run fast, locally, without requiring external network calls or heavy GPU reprocessing.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import FakeEmbeddings

from backend.app.rag.pipeline import (
    REQUIRED_JSONL_KEYS,
    ChunkRecord,
    IndexValidationError,
    chunk_document_pages,
    export_documents_jsonl,
    extract_pdf_pages,
    is_meaningful_text,
    load_and_validate_index,
    match_source_metadata,
    normalize_text,
    search_index,
    validate_documents_jsonl,
)


# ---------------------------------------------------------------------------
# 1. Text Normalization & Empty Text Handling
# ---------------------------------------------------------------------------

def test_text_normalization_preserves_code_and_math():
    """Verify whitespace normalization preserves python indentation and math formulations."""
    sample_raw = (
        "def compute_attention(q, k, v):\n"
        "    scores = q @ k.T\n"
        "    return   softmax(scores) @ v\n\n\n\n"
        "Loss formulation:   $$\\mathcal{L}  =  -\\sum y \\log(p)$$\n"
    )

    normalized = normalize_text(sample_raw)

    # Indentation of 4 spaces must be preserved
    assert "    scores = q @ k.T" in normalized
    assert "    return softmax(scores) @ v" in normalized
    # Math formulation preserved and internal spaces normalized
    assert "$$\\mathcal{L} = -\\sum y \\log(p)$$" in normalized
    # Excessive 4 newlines collapsed to 2
    assert "\n\n\n" not in normalized


def test_empty_page_handling_and_meaningful_text():
    """Verify empty pages, whitespace-only text, and non-alphanumeric text are detected."""
    assert not is_meaningful_text("")
    assert not is_meaningful_text("   \n\t   ")
    assert not is_meaningful_text("\x0c\x00")
    assert not is_meaningful_text("... --- ///")
    assert is_meaningful_text("Chapter 1: Deep Learning")
    assert is_meaningful_text("x = 42")


def test_empty_pages_do_not_produce_empty_chunks():
    """Verify that empty pages do not generate empty or whitespace-only chunks."""
    valid_pages = []  # No valid pages
    source_meta = {
        "source_id": "test_empty",
        "source_title": "Empty Book",
        "file_name": "empty.pdf",
        "relative_path": "data/rag/raw/empty.pdf",
        "subject": "math",
        "source_url": None,
    }

    chunks = chunk_document_pages(valid_pages, source_meta)
    assert len(chunks) == 0


# ---------------------------------------------------------------------------
# 2. Chunking Strategy & Page Provenance
# ---------------------------------------------------------------------------

def test_chunking_word_target_and_overlap():
    """Verify configurable chunking with word-count length function produces expected sizes."""
    # Generate ~600 words of synthetic text across 2 pages
    words_p1 = ["word"] * 300
    words_p2 = ["term"] * 300
    valid_pages = [
        (1, " ".join(words_p1)),
        (2, " ".join(words_p2)),
    ]

    source_meta = {
        "source_id": "test_book",
        "source_title": "Test ML Textbook",
        "file_name": "test_ml.pdf",
        "relative_path": "data/rag/raw/test/test_ml.pdf",
        "subject": "machine_learning",
        "source_url": "https://example.com/test",
    }

    chunks = chunk_document_pages(
        valid_pages=valid_pages,
        source_metadata=source_meta,
        chunk_size=220,
        chunk_overlap=40,
    )

    assert len(chunks) >= 3
    for chunk in chunks:
        word_count = len(chunk.text.split())
        # All chunks must be non-empty and around target size
        assert 10 <= word_count <= 250
        assert chunk.source_id == "test_book"
        assert chunk.source_title == "Test ML Textbook"
        assert len(chunk.content_hash) == 64  # SHA-256


def test_page_provenance_crossing_page_boundaries():
    """Verify that chunks crossing page boundaries accurately track page_start and page_end."""
    # Short pages so that a 220-word chunk spans across page 1 and page 2
    text_p1 = "Introduction to deep neural networks. " * 5  # ~25 words
    text_p2 = "Convolutional layers perform spatial feature extraction. " * 5  # ~30 words
    text_p3 = "Transformers utilize self-attention mechanisms for sequences. " * 40  # ~280 words

    valid_pages = [
        (1, text_p1.strip()),
        (2, text_p2.strip()),
        (3, text_p3.strip()),
    ]

    source_meta = {
        "source_id": "dl_foundations",
        "source_title": "Deep Learning Foundations",
        "file_name": "dl_foundations.pdf",
        "relative_path": "data/rag/raw/deep_learning/dl_foundations.pdf",
        "subject": "deep_learning",
        "source_url": None,
    }

    chunks = chunk_document_pages(
        valid_pages=valid_pages,
        source_metadata=source_meta,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) >= 2
    # The first chunk should span across page 1 and page 2
    first_chunk = chunks[0]
    assert first_chunk.page_start == 1
    assert first_chunk.page_end in (1, 2)

    # All chunks must have valid ordered page ranges
    for c in chunks:
        assert 1 <= c.page_start <= c.page_end <= 3


def test_deterministic_chunking_reproducibility():
    """Verify that identical input text produces identical chunk IDs, hashes, and text."""
    valid_pages = [
        (1, "Linear algebra is the study of vectors and linear mappings. " * 20),
        (2, "Matrix decomposition methods include SVD and eigenvalue decomposition. " * 20),
    ]
    source_meta = {
        "source_id": "math_foundations",
        "source_title": "Mathematics for ML",
        "file_name": "mml.pdf",
        "relative_path": "data/rag/raw/math/mml.pdf",
        "subject": "math",
        "source_url": None,
    }

    run1 = chunk_document_pages(valid_pages, source_meta, chunk_size=150, chunk_overlap=30)
    run2 = chunk_document_pages(valid_pages, source_meta, chunk_size=150, chunk_overlap=30)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.content_hash == c2.content_hash
        assert c1.page_start == c2.page_start
        assert c1.page_end == c2.page_end


# ---------------------------------------------------------------------------
# 3. Metadata Enrichment & sources.json Matching
# ---------------------------------------------------------------------------

def test_metadata_enrichment_exact_match(tmp_path):
    """Verify that matching sources.json enriches title, verified URL, and subject."""
    sources_manifest = {
        "d2l": {
            "source_id": "d2l",
            "title": "Dive into Deep Learning",
            "source_url": "https://d2l.ai/",
            "subject": "deep_learning",
        }
    }

    fake_pdf = tmp_path / "d2l" / "dive_into_deep_learning.pdf"
    fake_pdf.parent.mkdir(parents=True)
    fake_pdf.write_text("dummy", encoding="utf-8")

    meta = match_source_metadata(
        pdf_path=fake_pdf,
        project_root=tmp_path,
        sources_manifest=sources_manifest,
        pdf_title="",
    )

    assert meta["source_id"] == "d2l"
    assert meta["source_title"] == "Dive into Deep Learning"
    assert meta["source_url"] == "https://d2l.ai/"
    assert meta["subject"] == "deep_learning"
    assert meta["file_name"] == "dive_into_deep_learning.pdf"


def test_metadata_enrichment_unmatched_sets_url_to_none(tmp_path):
    """Verify that unverified sources do not hallucinate URLs and set source_url to None."""
    sources_manifest = {}  # Empty manifest

    fake_pdf = tmp_path / "math" / "probabilistic_machine_learning.pdf"
    fake_pdf.parent.mkdir(parents=True)
    fake_pdf.write_text("dummy", encoding="utf-8")

    meta = match_source_metadata(
        pdf_path=fake_pdf,
        project_root=tmp_path,
        sources_manifest=sources_manifest,
        pdf_title="Probabilistic Machine Learning",
    )

    assert meta["source_id"] == "probabilistic_machine_learning"
    assert meta["source_title"] == "Probabilistic Machine Learning"
    assert meta["source_url"] is None
    assert meta["subject"] == "math"


# ---------------------------------------------------------------------------
# 4. JSONL Export & Schema Validation
# ---------------------------------------------------------------------------

def test_documents_jsonl_export_and_validation(tmp_path):
    """Verify exporting ChunkRecords writes valid UTF-8 JSONL conforming to required schema."""
    chunks = [
        ChunkRecord(
            chunk_id=f"doc_chunk_{i:05d}",
            source_id="doc",
            source_title="Document Title",
            file_name="doc.pdf",
            relative_path="data/rag/raw/doc.pdf",
            page_start=i,
            page_end=i + 1,
            subject="nlp",
            source_url="https://example.com/doc" if i % 2 == 0 else None,
            text=f"This is chunk {i} discussing attention and transformers.",
            content_hash=f"hash_{i:04d}",
        )
        for i in range(1, 6)
    ]

    out_file = tmp_path / "documents.jsonl"
    export_documents_jsonl(chunks, out_file)

    assert out_file.exists()
    count = validate_documents_jsonl(out_file, expected_count=5)
    assert count == 5

    # Check that each line is independent JSON with all 11 required keys
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            assert REQUIRED_JSONL_KEYS.issubset(data.keys())


def test_documents_jsonl_validation_fails_on_corrupt_data(tmp_path):
    """Verify validate_documents_jsonl raises ValueError on invalid schema."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"chunk_id": "1", "text": "only two fields"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required keys"):
        validate_documents_jsonl(bad_file)


# ---------------------------------------------------------------------------
# 5. FAISS Index Alignment & Ordering
# ---------------------------------------------------------------------------

def test_faiss_vectorstore_ordering_and_manifest(tmp_path):
    """Verify that FAISS row ordering maps exactly 1-to-1 to input ChunkRecords."""
    fake_embeddings = FakeEmbeddings(size=384)

    chunks = [
        ChunkRecord(
            chunk_id=f"test_{i:05d}",
            source_id="test",
            source_title="Test Title",
            file_name="test.pdf",
            relative_path="data/rag/raw/test.pdf",
            page_start=i,
            page_end=i,
            subject="deep_learning",
            source_url=None,
            text=f"Vector index test chunk number {i}",
            content_hash=f"hash_{i}",
        )
        for i in range(10)
    ]

    texts = [c.text for c in chunks]
    metadatas = [c.to_dict() for c in chunks]

    vs = FAISS.from_texts(
        texts=texts,
        embedding=fake_embeddings,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    assert vs.index.ntotal == 10
    # Check 1-to-1 index-to-docstore ordering
    for idx in range(10):
        doc_id = vs.index_to_docstore_id[idx]
        doc = vs.docstore.search(doc_id)
        assert doc.metadata["chunk_id"] == f"test_{idx:05d}"
        assert doc.page_content == f"Vector index test chunk number {idx}"

    # Test saving local
    vs.save_local(str(tmp_path / "faiss_index"))
    assert (tmp_path / "faiss_index" / "index.faiss").exists()
    assert (tmp_path / "faiss_index" / "index.pkl").exists()


# ---------------------------------------------------------------------------
# 6. Missing Files & Index Validation Error Handling
# ---------------------------------------------------------------------------

def test_search_validation_missing_files_error(tmp_path):
    """Verify that load_and_validate_index raises IndexValidationError when index files are missing."""
    empty_dir = tmp_path / "empty_index"
    empty_dir.mkdir()

    with pytest.raises(IndexValidationError, match="FAISS index file missing"):
        load_and_validate_index(index_dir=empty_dir, model_path=tmp_path)


def test_search_validation_corrupted_manifest_error(tmp_path):
    """Verify that load_and_validate_index raises IndexValidationError when manifest is corrupted."""
    idx_dir = tmp_path / "bad_index"
    idx_dir.mkdir()
    (idx_dir / "index.faiss").write_text("fake_faiss", encoding="utf-8")
    (idx_dir / "index.pkl").write_text("fake_pkl", encoding="utf-8")
    (idx_dir / "index_manifest.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(IndexValidationError, match="Failed to parse index manifest"):
        load_and_validate_index(index_dir=idx_dir, model_path=tmp_path)


def test_search_index_results_formatting(tmp_path):
    """Verify search_index correctly formats SearchResult with score and snippets."""
    fake_embeddings = FakeEmbeddings(size=384)
    texts = ["Backpropagation computes the gradient of the loss with respect to weights."]
    metadatas = [{
        "chunk_id": "bp_00001",
        "source_id": "dl_book",
        "source_title": "Deep Learning Book",
        "file_name": "dl.pdf",
        "page_start": 42,
        "page_end": 43,
        "subject": "deep_learning",
        "source_url": "https://example.com/dl",
    }]

    vs = FAISS.from_texts(
        texts=texts,
        embedding=fake_embeddings,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    results = search_index(vs, query="gradient of the loss", top_k=1)
    assert len(results) == 1
    res = results[0]
    assert res.rank == 1
    assert res.chunk_id == "bp_00001"
    assert res.source_title == "Deep Learning Book"
    assert res.file_name == "dl.pdf"
    assert res.page_start == 42
    assert res.page_end == 43
    assert "Backpropagation" in res.snippet


# ---------------------------------------------------------------------------
# 7. Real-Content Regression Tests for PDF Page Provenance
# ---------------------------------------------------------------------------

def test_regression_d2l_backpropagation_page_provenance():
    """
    Regression test: The chunk containing '5.3.3 Backpropagation' in the D2L PDF
    must NOT report pages 1-4, but rather the actual physical page (~221).
    Derives the expected physical PDF page dynamically from extracted text.
    """
    import fitz
    pdf_path = Path("data/rag/raw/d2l/dive_into_deep_learning.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Corpus PDF not found at {pdf_path}")

    # Derive expected physical PDF page (1-indexed, skipping initial TOC)
    doc = fitz.open(pdf_path)
    expected_page = next(
        idx + 1 for idx, p in enumerate(doc)
        if idx > 20 and "5.3.3" in p.get_text() and "Backpropagation" in p.get_text()
    )

    valid_pages, _, meta = extract_pdf_pages(pdf_path)
    source_meta = match_source_metadata(pdf_path, pdf_title=meta.get("title"))

    chunks = chunk_document_pages(valid_pages, source_meta, chunk_size=220, chunk_overlap=40)
    # Target the chapter content chunk containing section 5.3.3 and gradient explanation (outside TOC)
    matching_chunks = [
        c for c in chunks
        if "5.3.3" in c.text and "Backpropagation" in c.text and "gradient" in c.text
    ]
    assert len(matching_chunks) > 0, "No content chunk found containing 5.3.3 Backpropagation"

    for chunk in matching_chunks:
        # Crucial bug check: MUST NOT report pages 1-4
        assert not (chunk.page_start == 1 and chunk.page_end <= 4), (
            f"Bug reproduced: chunk {chunk.chunk_id} incorrectly reported pages {chunk.page_start}-{chunk.page_end}"
        )
        assert chunk.page_start <= expected_page <= chunk.page_end, (
            f"Chunk {chunk.chunk_id} page range ({chunk.page_start}-{chunk.page_end}) "
            f"does not include expected physical page {expected_page}"
        )


def test_regression_slp_equation_6_31_page_provenance():
    """
    Regression test: The chunk containing equation '(6.31)' in Speech and Language Processing
    must NOT report pages 1-2, but rather the actual physical page (~179).
    Derives the expected physical PDF page dynamically from extracted text.
    """
    import fitz
    pdf_path = Path("data/rag/raw/nlp_llm/speech_and_language_processing.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Corpus PDF not found at {pdf_path}")

    # Derive expected physical PDF page (1-indexed)
    doc = fitz.open(pdf_path)
    expected_page = next(
        idx + 1 for idx, p in enumerate(doc)
        if "(6.31)" in p.get_text()
    )

    valid_pages, _, meta = extract_pdf_pages(pdf_path)
    source_meta = match_source_metadata(pdf_path, pdf_title=meta.get("title"))

    chunks = chunk_document_pages(valid_pages, source_meta, chunk_size=220, chunk_overlap=40)
    matching_chunks = [c for c in chunks if "(6.31)" in c.text]
    assert len(matching_chunks) > 0, "No chunk found containing equation (6.31)"

    for chunk in matching_chunks:
        # Crucial bug check: MUST NOT report pages 1-2
        assert not (chunk.page_start == 1 and chunk.page_end <= 2), (
            f"Bug reproduced: chunk {chunk.chunk_id} incorrectly reported pages {chunk.page_start}-{chunk.page_end}"
        )
        assert chunk.page_start <= expected_page <= chunk.page_end, (
            f"Chunk {chunk.chunk_id} page range ({chunk.page_start}-{chunk.page_end}) "
            f"does not include expected physical page {expected_page}"
        )


def test_regression_repeated_text_across_pages_monotonicity():
    """Verify repeated phrases across distinct pages are mapped to their correct sequential physical pages."""
    valid_pages = [
        (10, "Summary of chapter findings. " * 10),
        (20, "Intermediate discussion of neural networks. " * 10),
        (30, "Summary of chapter findings. " * 10),  # Identical phrase on page 30
    ]
    source_meta = {
        "source_id": "test_repeat",
        "source_title": "Repeated Text Test",
        "file_name": "test_repeat.pdf",
        "relative_path": "data/rag/raw/test_repeat.pdf",
        "subject": "testing",
        "source_url": None,
    }

    chunks = chunk_document_pages(valid_pages, source_meta, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 3

    # First chunk containing summary must be on page 10
    assert chunks[0].page_start == 10
    # Last chunk containing summary must be on page 30, NOT reset to page 10
    assert chunks[-1].page_start == 30

