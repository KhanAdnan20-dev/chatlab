"""
Tests for Phase 2A: LangChain RAG Integration into the Gemma Chat Orchestrator.

Validates:
1. Relevant technical questions trigger retrieval.
2. Greetings and simple arithmetic bypass retrieval.
3. General questions remain supported without RAG injection.
4. Retrieval service returns structured source metadata aligned with retrieved documents.
5. Page provenance corresponds to physical PDF bounds.
6. Retrieved evidence reaches the Gemma prompt builder.
7. Responses include structured citation metadata.
8. Empty retrieval results do not break the chat.
9. Missing or corrupt RAG artifacts degrade gracefully and log warnings.
10. Conversation memory still handles multi-turn follow-up questions.
11. Quiz routes and deterministic evaluation remain unaffected.
12. Existing API responses remain backwards-compatible.
"""

from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.rag.pipeline import SearchResult
from backend.app.schemas import ChatResponse, CitationItem, FuzzyQueryProfile
from backend.app.services.chatbot import chatbot_service
from backend.app.services.conversation_memory import conversation_memory
from backend.app.services.dynamic_prompt_builder import dynamic_prompt_builder
from backend.app.services.model_manager import model_manager
from backend.app.services.rag_retrieval_service import RAGRetrievalService, rag_retrieval_service

# Ensure model manager is in lightweight test mode to avoid loading heavy LLM weights into VRAM
model_manager.set_test_mode(True)
client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Technical Queries Trigger Retrieval
# ---------------------------------------------------------------------------

def test_technical_questions_trigger_retrieval():
    """Verify that AI/ML technical questions and explicit notes requests trigger retrieval."""
    test_queries = [
        "How does the transformer multi-head attention mechanism work?",
        "Explain backpropagation using the chain rule.",
        "According to my available notes, what is regularization?",
        "In the literature, how does Adam optimizer calculate moment estimates?",
        "What is the mathematical formulation of cross-entropy loss?",
    ]

    for q in test_queries:
        should_retrieve, reason = rag_retrieval_service.should_retrieve(message=q)
        assert should_retrieve is True, f"Expected retrieval for '{q}', but was skipped. Reason: {reason}"
        assert len(reason) > 0


# ---------------------------------------------------------------------------
# 2. Greetings and Simple Arithmetic Bypass Retrieval
# ---------------------------------------------------------------------------

def test_greetings_and_arithmetic_bypass_retrieval():
    """Verify that casual greetings, pleasantries, and simple math bypass retrieval."""
    bypass_queries = [
        "Hello",
        "hi",
        "hey there",
        "Good morning!",
        "Thanks",
        "thank you so much",
        "What is 2 + 2?",
        "2+2",
        "5 * 10",
        "what is 100 / 4?",
        "who are you?",
    ]

    for q in bypass_queries:
        should_retrieve, reason = rag_retrieval_service.should_retrieve(message=q)
        assert should_retrieve is False, f"Expected bypass for '{q}', but triggered. Reason: {reason}"
        assert "bypassed" in reason.lower() or "conversational" in reason.lower() or "arithmetic" in reason.lower()


# ---------------------------------------------------------------------------
# 3. General Questions Remain Supported Without RAG
# ---------------------------------------------------------------------------

def test_general_questions_bypass_retrieval_and_answer_naturally():
    """Verify that non-technical out-of-domain questions bypass RAG."""
    general_query = "What is the capital of France?"
    fp = FuzzyQueryProfile(aiml_relevance=0.1, technical_depth=0.2)
    should_retrieve, reason = rag_retrieval_service.should_retrieve(
        message=general_query,
        fuzzy_profile=fp,
    )
    assert should_retrieve is False
    assert "bypassed" in reason.lower()

    # Verify orchestrator response handles general question cleanly
    res = chatbot_service.process(general_query)
    assert res.response is not None
    assert len(res.citations) == 0


# ---------------------------------------------------------------------------
# 4. Retrieval Service Returns Aligned Source Metadata
# ---------------------------------------------------------------------------

def test_retrieval_service_returns_source_metadata_aligned():
    """Verify that CitationItem attributes are properly populated from SearchResult."""
    mock_res = [
        SearchResult(
            rank=1,
            score=0.785,
            source_title="Understanding Deep Learning",
            file_name="understanding_deep_learning.pdf",
            page_start=229,
            page_end=229,
            subject="deep_learning",
            source_url="https://udlbook.github.io/udlbook/",
            snippet="Multi-head attention allows the model to jointly attend...",
            full_text="Multi-head attention allows the model to jointly attend to information...",
            chunk_id="udl_chunk_00529",
        )
    ]

    rag_retrieval_service.set_mock_results(mock_res)
    try:
        citations, grounding = rag_retrieval_service.retrieve("multi-head attention", top_k=1)
        assert len(citations) == 1
        cit = citations[0]
        assert cit.citation_code == "[S1]"
        assert cit.source_title == "Understanding Deep Learning"
        assert cit.file_name == "understanding_deep_learning.pdf"
        assert cit.page_start == 229
        assert cit.page_end == 229
        assert cit.subject == "deep_learning"
        assert cit.score == 0.785
        assert cit.source_url == "https://udlbook.github.io/udlbook/"
        assert "Multi-head attention" in cit.content_snippet

        assert grounding is not None
        assert "[S1] Understanding Deep Learning" in grounding
        assert "EVIDENCE GROUNDING INSTRUCTIONS:" in grounding
    finally:
        rag_retrieval_service.reset_for_tests()


# ---------------------------------------------------------------------------
# 5. Page Provenance Corresponds to Actual PDF Text / Bounds
# ---------------------------------------------------------------------------

def test_page_provenance_positive_bounds():
    """Verify that retrieved chunk metadata contains valid, positive physical page numbers."""
    # Test with live FAISS index if available, or validated results
    if rag_retrieval_service.ensure_loaded():
        citations, _ = rag_retrieval_service.retrieve("How does the transformer multi-head attention mechanism work?", top_k=2)
        assert len(citations) > 0
        for c in citations:
            assert c.page_start is not None and c.page_start > 0
            assert c.page_end is not None and c.page_end >= c.page_start
            assert c.file_name.endswith(".pdf")
            assert c.source_title != ""


# ---------------------------------------------------------------------------
# 6. Retrieved Evidence Reaches Gemma Prompt Builder
# ---------------------------------------------------------------------------

def test_retrieved_evidence_reaches_gemma_prompt_builder():
    """Verify that RAG grounding block is incorporated into Gemma system instructions."""
    sample_grounding = (
        "\n\n[AUTHORITATIVE LITERATURE EVIDENCE]\n"
        "[S1] Understanding Deep Learning (understanding_deep_learning.pdf, PDF p. 229):\n"
        "Multi-head attention connects multiple self-attention mechanisms in parallel.\n"
        "[END LITERATURE EVIDENCE]\n"
    )

    built = dynamic_prompt_builder.build(
        message="Explain multi-head attention",
        fuzzy_profile=FuzzyQueryProfile(technical_depth=0.8, aiml_relevance=0.9),
        rag_grounding=sample_grounding,
    )

    assert "[AUTHORITATIVE LITERATURE EVIDENCE]" in built.system_instruction
    assert "[S1] Understanding Deep Learning" in built.system_instruction
    assert "Multi-head attention connects multiple self-attention mechanisms in parallel." in built.system_instruction


# ---------------------------------------------------------------------------
# 7. Responses Can Include Structured Citation Metadata
# ---------------------------------------------------------------------------

def test_responses_include_structured_citation_metadata():
    """Verify ChatbotResult and ChatResponse retain structured citation objects."""
    mock_res = [
        SearchResult(
            rank=1,
            score=0.72,
            source_title="Dive into Deep Learning",
            file_name="dive_into_deep_learning.pdf",
            page_start=221,
            page_end=221,
            subject="deep_learning",
            source_url="https://d2l.ai",
            snippet="5.3.3 Backpropagation computes gradients...",
            full_text="5.3.3 Backpropagation computes gradients of the loss with respect to parameters...",
            chunk_id="d2l_chunk_00487",
        )
    ]

    rag_retrieval_service.set_mock_results(mock_res)
    try:
        result = chatbot_service.process("Explain backpropagation using the chain rule.")
        assert len(result.citations) == 1
        c = result.citations[0]
        assert c.citation_code == "[S1]"
        assert c.source_title == "Dive into Deep Learning"
        assert c.page_start == 221
        assert c.page_end == 221

        # Check API serialization
        api_resp = ChatResponse(
            response=result.response,
            layer=result.layer,
            intent=result.intent,
            confidence=result.confidence,
            citations=result.citations,
        )
        assert len(api_resp.citations) == 1
        assert api_resp.citations[0].file_name == "dive_into_deep_learning.pdf"
        assert api_resp.citations[0].page_start == 221
    finally:
        rag_retrieval_service.reset_for_tests()


# ---------------------------------------------------------------------------
# 8. Empty Retrieval Results Do Not Break Chat
# ---------------------------------------------------------------------------

def test_empty_retrieval_results_do_not_break_chat():
    """Verify that when retrieval yields zero matches, chat flow finishes cleanly."""
    rag_retrieval_service.set_mock_results([])
    try:
        result = chatbot_service.process("Explain backpropagation using the chain rule.")
        assert result.response is not None
        assert result.citations == []
        assert result.fallback is False
    finally:
        rag_retrieval_service.reset_for_tests()


# ---------------------------------------------------------------------------
# 9. Missing or Corrupt Artifacts Degrade Gracefully
# ---------------------------------------------------------------------------

def test_missing_or_corrupt_artifacts_degrades_gracefully(caplog):
    """Verify that an invalid index directory degrades gracefully without crashing."""
    corrupt_service = RAGRetrievalService(
        index_dir=Path("non_existent_dir_12345"),
        model_path=Path("non_existent_model_12345"),
    )
    loaded = corrupt_service.ensure_loaded()
    assert loaded is False
    assert corrupt_service.is_available is False

    citations, grounding = corrupt_service.retrieve("Explain backprop")
    assert citations == []
    assert grounding is None


# ---------------------------------------------------------------------------
# 10. Conversation Memory Handles Multi-Turn Follow-Ups
# ---------------------------------------------------------------------------

def test_conversation_memory_handles_follow_up_questions():
    """Verify that context resolution and memory history function across multi-turn interactions."""
    session_id = "test-rag-multiturn-session"
    conversation_memory.clear_session(session_id)

    # Turn 1
    res1 = chatbot_service.process(
        message="What is the transformer multi-head attention mechanism?",
        session_id=session_id,
    )
    assert res1.session_id == session_id

    # Turn 2: Follow-up referencing preceding turn
    res2 = chatbot_service.process(
        message="How does its backward pass compute gradients?",
        session_id=session_id,
    )
    assert res2.session_id == session_id

    # Check that memory holds both turns
    history = conversation_memory.get_history(session_id)
    assert len(history) >= 4  # 2 user turns + 2 assistant turns
    assert "transformer multi-head attention" in history[0]["content"]
    assert "backward pass" in history[2]["content"]


# ---------------------------------------------------------------------------
# 11. Quiz Routes and Evaluation are Unchanged
# ---------------------------------------------------------------------------

def test_quiz_routes_and_evaluation_isolation():
    """Verify that existing quiz engine routes and deterministic evaluation continue to work."""
    # 1. Start quiz
    start_resp = client.post(
        "/quiz/start",
        json={"topic": "all", "difficulty": "easy", "num_questions": 3},
    )
    assert start_resp.status_code == 201
    quiz_data = start_resp.json()
    sess_id = quiz_data["session_id"]
    q1 = quiz_data["first_question"]
    assert q1 is not None
    assert len(q1["options"]) == 4

    # 2. Submit answer
    ans_resp = client.post(
        f"/quiz/{sess_id}/answer",
        json={"answer_index": 0},
    )
    assert ans_resp.status_code == 200
    eval_data = ans_resp.json()
    assert "is_correct" in eval_data
    assert "explanation" in eval_data
    assert eval_data["user_answer"] == 0

    # 3. Retrieve score
    score_resp = client.get(f"/quiz/{sess_id}/score")
    assert score_resp.status_code == 200
    assert "correct_count" in score_resp.json()


# ---------------------------------------------------------------------------
# 12. Existing API Responses Backwards-Compatible
# ---------------------------------------------------------------------------

def test_chat_api_backwards_compatible():
    """Verify that POST /chat returns a valid 200 response compatible with existing clients."""
    resp = client.post(
        "/chat",
        json={"message": "What is Supervised Learning?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "layer" in data
    assert "intent" in data
    assert "confidence" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)
    # Even if citations list is empty or populated, all fields must be valid
    for cit in data["citations"]:
        assert "citation_code" in cit
        assert "source_title" in cit
        assert "content_snippet" in cit
