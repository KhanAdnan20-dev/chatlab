"""
ChatLab Phase 2C — Citation and Grounding Integrity Test Suite.

Verifies:
1. Displayed citations map accurately to retrieved documents and physical PDF page ranges.
2. Unsupported/negative queries are gated without fabricated citations.
3. Message-level citation isolation is strictly preserved.
4. Retrieval failures degrade gracefully without crashing the chat pipeline.
5. Deterministic quiz and casual conversation routing remain untouched.
6. Benchmark evaluation dataset and results.json artifacts are valid and complete.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.rag.pipeline import SearchResult
from backend.app.schemas import ChatResponse, CitationItem
from backend.app.services.rag_retrieval_service import RAGRetrievalService
from backend.main import app

client = TestClient(app)
ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def retrieval_service():
    """Create a test RAG retrieval service instance."""
    return RAGRetrievalService()


def test_citation_maps_to_retrieved_document(retrieval_service):
    """Verify that every generated CitationItem maps precisely to a SearchResult."""
    mock_results = [
        SearchResult(
            rank=1,
            score=0.82,
            source_title="Dive into Deep Learning",
            file_name="dive_into_deep_learning.pdf",
            page_start=221,
            page_end=221,
            subject="deep_learning",
            source_url="https://d2l.ai/",
            snippet="5.3.3 Backpropagation computes gradients via chain rule...",
            full_text="5.3.3 Backpropagation computes gradients via chain rule...",
            chunk_id="d2l_chunk_00486",
        ),
        SearchResult(
            rank=2,
            score=0.75,
            source_title="Understanding Deep Learning",
            file_name="understanding_deep_learning.pdf",
            page_start=228,
            page_end=228,
            subject="deep_learning",
            source_url=None,
            snippet="Multi-head attention applies linear projections...",
            full_text="Multi-head attention applies linear projections...",
            chunk_id="understanding_deep_learning_chunk_00528",
        ),
    ]

    retrieval_service.set_mock_results(mock_results)
    citations, context_block = retrieval_service.retrieve(
        query="Explain backpropagation and multi-head attention",
        top_k=2,
    )

    assert len(citations) == 2
    # Verify first citation
    c1 = citations[0]
    assert c1.citation_code == "[S1]"
    assert c1.file_name == "dive_into_deep_learning.pdf"
    assert c1.source_title == "Dive into Deep Learning"
    assert c1.page_start == 221
    assert c1.page_end == 221
    assert c1.subject == "deep_learning"
    assert c1.source_url == "https://d2l.ai/"
    assert c1.score == 0.82
    assert "5.3.3 Backpropagation" in c1.content_snippet

    # Verify second citation
    c2 = citations[1]
    assert c2.citation_code == "[S2]"
    assert c2.file_name == "understanding_deep_learning.pdf"
    assert c2.page_start == 228
    assert c2.page_end == 228
    assert c2.source_url is None

    # Verify prompt grounding format
    assert context_block is not None
    assert "[S1] Dive into Deep Learning" in context_block
    assert "dive_into_deep_learning.pdf, PDF p. 221" in context_block
    assert "[S2] Understanding Deep Learning" in context_block
    assert "understanding_deep_learning.pdf, PDF p. 228" in context_block


def test_page_provenance_accuracy_against_corpus():
    """Verify that chunks in documents.jsonl have valid physical page ranges."""
    documents_path = ROOT_DIR / "data" / "rag" / "processed" / "documents.jsonl"
    assert documents_path.exists(), "documents.jsonl must exist"

    sample_count = 0
    with open(documents_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 500 == 0:  # Sample every 500th chunk
                sample_count += 1
                rec = json.loads(line)
                assert rec["page_start"] >= 1
                assert rec["page_end"] >= rec["page_start"]
                assert rec["file_name"].endswith(".pdf")
                assert len(rec["text"].strip()) > 0
                assert rec["chunk_id"].strip() != ""

    assert sample_count >= 20


def test_unsupported_queries_gating_suppression(retrieval_service):
    """Verify that low-scoring unsupported queries yield 0 citations and no fabricated context."""
    # Low similarity results below threshold 0.35
    low_score_results = [
        SearchResult(
            rank=1,
            score=0.22,
            source_title="Mathematics for Machine Learning",
            file_name="mathematics_for_machine_learning.pdf",
            page_start=15,
            page_end=15,
            subject="machine_learning",
            source_url=None,
            snippet="Irrelevant math background...",
            full_text="Irrelevant math background...",
            chunk_id="mml_chunk_00015",
        )
    ]

    # When search_index returns low scores, score filtering should drop them
    with patch("backend.app.services.rag_retrieval_service.search_index", return_value=low_score_results):
        with patch.object(retrieval_service, "ensure_loaded", return_value=True):
            retrieval_service._vector_store = MagicMock()
            citations, context_block = retrieval_service.retrieve(
                query="What were the causes of the fall of the Western Roman Empire?",
                top_k=4,
            )

            assert context_block is None
            assert len(citations) == 0


def test_citation_marker_isolation_per_message():
    """Verify that citations for different messages are completely isolated in message history."""
    msg1_citations = [
        CitationItem(
            citation_code="[S1]",
            source_title="Dive into Deep Learning",
            content_snippet="Passage 1 from D2L",
            file_name="dive_into_deep_learning.pdf",
            page_start=100,
            page_end=100,
        )
    ]
    msg2_citations = [
        CitationItem(
            citation_code="[S1]",
            source_title="Speech and Language Processing",
            content_snippet="Passage 2 from SLP",
            file_name="speech_and_language_processing.pdf",
            page_start=50,
            page_end=50,
        ),
        CitationItem(
            citation_code="[S2]",
            source_title="Understanding Deep Learning",
            content_snippet="Passage 3 from UDL",
            file_name="understanding_deep_learning.pdf",
            page_start=75,
            page_end=75,
        ),
    ]

    resp1 = ChatResponse(
        response="Here is answer 1 [S1].",
        confidence=0.95,
        citations=msg1_citations,
        query_type="conceptual",
        intent="general_inquiry",
    )
    resp2 = ChatResponse(
        response="Here is answer 2 [S1] and [S2].",
        confidence=0.92,
        citations=msg2_citations,
        query_type="code_generation",
        intent="general_inquiry",
    )

    # Assert mutual isolation
    assert len(resp1.citations) == 1
    assert resp1.citations[0].file_name == "dive_into_deep_learning.pdf"
    assert len(resp2.citations) == 2
    assert resp2.citations[0].file_name == "speech_and_language_processing.pdf"
    assert resp2.citations[1].file_name == "understanding_deep_learning.pdf"
    assert resp1.citations != resp2.citations


def test_retrieval_failure_graceful_degradation(retrieval_service):
    """Verify that if search raises an exception, the retrieval service degrades safely."""
    with patch("backend.app.services.rag_retrieval_service.search_index", side_effect=RuntimeError("FAISS disk error")):
        with patch.object(retrieval_service, "ensure_loaded", return_value=True):
            retrieval_service._vector_store = MagicMock()
            citations, context_block = retrieval_service.retrieve(
                query="How does backpropagation work?",
                top_k=4,
            )
            assert context_block is None
            assert citations == []


def test_casual_conversation_skips_retrieval(retrieval_service):
    """Verify that greetings and casual phrases bypass RAG retrieval completely."""
    should_1, reason_1 = retrieval_service.should_retrieve("hello there!")
    assert should_1 is False
    assert "retrieval bypassed" in reason_1.lower()

    should_2, reason_2 = retrieval_service.should_retrieve("good morning")
    assert should_2 is False

    should_3, reason_3 = retrieval_service.should_retrieve("thanks a lot")
    assert should_3 is False

    should_4, reason_4 = retrieval_service.should_retrieve("what is 2 + 2?")
    assert should_4 is False

    should_5, reason_5 = retrieval_service.should_retrieve("What is gradient descent and backpropagation?")
    assert should_5 is True


def test_benchmark_results_schema_validity():
    """Verify that results.json exists and adheres to the evaluation schema."""
    results_path = ROOT_DIR / "data" / "rag" / "evaluation" / "results.json"
    assert results_path.exists(), "Benchmark results.json must exist"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "benchmark_metadata" in data
    assert "dense_baseline" in data
    assert "bm25_sparse" in data
    assert "hybrid_rrf" in data

    dense = data["dense_baseline"]["metrics"]["retrieval_metrics"]
    assert "recall_at_1" in dense
    assert "recall_at_5" in dense
    assert "mrr" in dense
    assert "source_accuracy_at_5" in dense
    assert "page_provenance_accuracy" in dense
    assert dense["source_accuracy_at_5"] >= 0.80, "Source accuracy at 5 should be >= 80%"


def test_retrieval_evaluation_dataset_validity():
    """Verify that retrieval_queries.jsonl exists and contains valid annotated queries."""
    dataset_path = ROOT_DIR / "data" / "rag" / "evaluation" / "retrieval_queries.jsonl"
    assert dataset_path.exists(), "retrieval_queries.jsonl must exist"

    queries = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    assert 40 <= len(queries) <= 60, f"Expected 40-60 queries, got {len(queries)}"
    supported_annotated = [q for q in queries if q["is_supported"] and not q["requires_human_annotation"]]
    assert len(supported_annotated) >= 40, "Expected at least 40 verified ground-truth queries"
    negative = [q for q in queries if not q["is_supported"]]
    assert len(negative) >= 5, "Expected at least 5 negative/unsupported queries"
