"""Tests for Grounded RAG Knowledge Base, Roadmap Generator, Mastery Tracker, and Socratic Tutor."""
import pytest
from backend.app.services.mastery_state import (
    STATUS_IN_PROGRESS,
    STATUS_MASTERED,
    STATUS_NOT_STARTED,
    mastery_tracker,
)
from backend.app.services.rag_engine import rag_engine
from backend.app.services.roadmap_engine import roadmap_engine
from backend.app.services.socratic_tutor import socratic_tutor


def test_rag_retrieval_returns_relevant_literature():
    """Verify RAG engine indexes knowledge corpus and retrieves matches with citations."""
    results = rag_engine.retrieve("How does the transformer attention mechanism work?", top_k=2)
    assert len(results) > 0

    top_match = results[0]
    assert "Vaswani" in top_match.source or "Attention" in top_match.title
    assert top_match.citation.startswith("[")
    assert len(top_match.formulas) > 0


def test_rag_formatting_grounding_prompt():
    """Verify formatting generates verified grounding instructions with formulas."""
    results = rag_engine.retrieve("Explain backpropagation with chain rule", top_k=1)
    assert len(results) > 0
    prompt_addon = rag_engine.format_grounding_prompt(results)

    assert "[VERIFIED AI/ML KNOWLEDGE GROUNDING]" in prompt_addon
    assert "CITATION TAG:" in prompt_addon
    assert "Goodfellow" in prompt_addon or "Back-Propagation" in prompt_addon


def test_roadmap_domain_detection():
    """Verify domain detection maps natural language to curriculum domains."""
    assert roadmap_engine.detect_domain("give me an NLP roadmap") == "nlp_llms"
    assert roadmap_engine.detect_domain("deep learning curriculum") == "deep_learning"
    assert roadmap_engine.detect_domain("computer vision learning path") == "computer_vision"
    assert roadmap_engine.detect_domain("teach me everything about machine learning") == "classical_ml"
    assert roadmap_engine.detect_domain("full curriculum") == "all"


def test_roadmap_topological_sort_and_prerequisites():
    """Verify induced subgraph includes prerequisite ancestors in valid topological order."""
    nodes, edges = roadmap_engine.get_induced_subgraph("nlp_llms")
    node_ids = [n.id for n in nodes]

    # Attention & Transformers requires Linear Algebra and Calculus
    assert "attention_mech" in node_ids
    assert "lin_alg" in node_ids
    assert "calc_opt" in node_ids
    bp_id = "perceptron_backprop" if "perceptron_backprop" in node_ids else "backprop_mlp"
    assert bp_id in node_ids

    # Check topological order: prerequisites must precede target
    idx_lin_alg = node_ids.index("lin_alg")
    idx_backprop = node_ids.index(bp_id)
    idx_attention = node_ids.index("attention_mech")
    assert idx_lin_alg < idx_backprop < idx_attention


def test_roadmap_mermaid_generation_syntax_safety():
    """Verify generated Mermaid code is clean, syntax-safe, and contains quoted labels."""
    rd = roadmap_engine.generate_roadmap("give me an nlp and transformers roadmap")
    assert "graph TD" in rd.mermaid_code
    assert 'subgraph Phase_' in rd.mermaid_code
    assert rd.node_count > 5
    assert len(rd.phases) > 0
    assert "```mermaid" in rd.markdown_content

    # Every node in Mermaid must have quotes to prevent browser parser crashes
    for line in rd.mermaid_code.split("\n"):
        if '["' in line:
            assert '"]' in line


def test_mastery_tracker_progression_and_recommendation():
    """Verify student mastery updates and next recommended node logic."""
    session_id = "test-session-mastery-42"
    profile = mastery_tracker.get_profile(session_id)

    # Initial state
    assert profile.get_status("py_prog") == STATUS_NOT_STARTED

    # Discussed
    mastery_tracker.update_node(session_id, "py_prog", STATUS_IN_PROGRESS)
    assert profile.get_status("py_prog") == STATUS_IN_PROGRESS

    # Mastered via quiz
    mastery_tracker.update_node(session_id, "py_prog", STATUS_MASTERED, quiz_score=0.95)
    assert profile.get_status("py_prog") == STATUS_MASTERED

    # Next recommendation should now be lin_alg (since py_prog is mastered)
    rec = mastery_tracker.get_recommended_next(session_id)
    assert rec is not None
    assert rec.id == "lin_alg"


def test_socratic_tutor_directive_generation():
    """Verify Socratic tutor resolves concept prerequisites and constructs active recall prompts."""
    ctx = socratic_tutor.build_socratic_directive("Why do we need LayerNorm in transformers?")
    assert ctx.target_node is not None
    assert ctx.prompt_directive is not None
    assert "[SOCRATIC TUTOR PEDAGOGICAL MODE ACTIVATED]" in ctx.prompt_directive
    assert "DO NOT give away the complete final answer" in ctx.prompt_directive


def test_socratic_tutor_direct_instruction_override():
    """Verify that when a user asks to 'just explain it', Socratic tutor grants direct instruction."""
    ctx = socratic_tutor.build_socratic_directive("just explain it", history=[
        {"role": "user", "content": "Explain Attention Mechanisms & Transformers"},
        {"role": "assistant", "content": "What role do you think input elements need to play?"},
    ])
    assert "[SOCRATIC TUTOR PEDAGOGICAL MODE - DIRECT INSTRUCTION OVERRIDE]" in ctx.prompt_directive
    assert "Transition from questioning to comprehensive Direct Instruction" in ctx.prompt_directive


def test_socratic_tutor_continuation_handling():
    """Verify that when a user says 'continue', Socratic tutor resumes seamlessly without filler."""
    ctx = socratic_tutor.build_socratic_directive("continue", history=[
        {"role": "user", "content": "Explain Attention Mechanisms & Transformers"},
        {"role": "assistant", "content": "1. Query (Q)... 2. Keys (K)..."},
    ])
    assert "[SOCRATIC TUTOR PEDAGOGICAL MODE - SEAMLESS CONTINUATION]" in ctx.prompt_directive
    assert "Seamlessly resume the ongoing explanation" in ctx.prompt_directive


def test_socratic_tutor_surrender_handling():
    """Verify that when a user says 'idk', Socratic tutor provides an encouraging hint/analogy."""
    ctx = socratic_tutor.build_socratic_directive("idk", history=[
        {"role": "user", "content": "Explain Attention Mechanisms & Transformers"},
        {"role": "assistant", "content": "What mathematical property allows the model to score vectors?"},
    ])
    assert "[SOCRATIC TUTOR PEDAGOGICAL MODE - GUIDED SCAFFOLDING & HINT]" in ctx.prompt_directive
    assert "Provide an intuitive, real-world analogy" in ctx.prompt_directive


def test_socratic_prompt_builder_token_budget_override():
    """Verify that Socratic direct instruction override receives a generous token budget (>= 640)."""
    from backend.app.services.dynamic_prompt_builder import dynamic_prompt_builder
    from backend.app.schemas import FuzzyQueryProfile

    # 1. Direct instruction override in Socratic mode
    ctx_override = socratic_tutor.build_socratic_directive("just explain it")
    prompt_override = dynamic_prompt_builder.build(
        message="just explain it",
        fuzzy_profile=FuzzyQueryProfile(technical_depth=0.8),
        socratic_directive=ctx_override.prompt_directive,
    )
    assert prompt_override.token_budget >= 640

    # 2. Standard Socratic active recall prompt has comfortable room (>= 280 tokens)
    ctx_standard = socratic_tutor.build_socratic_directive("Explain Backpropagation")
    prompt_standard = dynamic_prompt_builder.build(
        message="Explain Backpropagation",
        fuzzy_profile=FuzzyQueryProfile(technical_depth=0.8),
        socratic_directive=ctx_standard.prompt_directive,
    )
    assert prompt_standard.token_budget >= 280

