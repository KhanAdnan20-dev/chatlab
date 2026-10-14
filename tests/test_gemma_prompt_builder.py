"""Tests for Dynamic Prompt Builder and Mamdani Fuzzy Profile Translation for Gemma 4."""
import pytest
from backend.app.schemas import FuzzyQueryProfile
from backend.app.services.dynamic_prompt_builder import dynamic_prompt_builder


def test_prompt_builder_eli5_query():
    fp = FuzzyQueryProfile(
        technical_depth=0.15,
        beginner_friendliness=0.92,
        conversationalness=0.90,
        mathematical_detail=0.10,
        example_preference=0.88,
        code_focus=0.10,
        aiml_relevance=0.85,
    )
    prompt = dynamic_prompt_builder.build(
        message="Teach me recursion like I'm five.",
        fuzzy_profile=fp,
    )

    assert "ChatLab" in prompt.system_instruction
    assert "simple analogies" in prompt.system_instruction or "beginner-friendly" in prompt.system_instruction
    assert "Avoid unnecessary jargon" in prompt.system_instruction
    assert len(prompt.messages) == 2  # system + user
    assert prompt.messages[1]["content"] == "Teach me recursion like I'm five."
    assert prompt.token_budget <= 640


def test_prompt_builder_deep_math_query():
    fp = FuzzyQueryProfile(
        technical_depth=0.88,
        beginner_friendliness=0.20,
        conversationalness=0.40,
        mathematical_detail=0.90,
        example_preference=0.30,
        code_focus=0.20,
        aiml_relevance=0.95,
        query_complexity=0.85,
        explanation_depth=0.85,
    )
    prompt = dynamic_prompt_builder.build(
        message="Derive the gradient of cross-entropy loss function mathematically.",
        fuzzy_profile=fp,
    )

    assert "rigorous technical depth" in prompt.system_instruction.lower()
    assert "formulas" in prompt.system_instruction.lower() or "mathematical" in prompt.system_instruction.lower()
    assert prompt.token_budget >= 192


def test_prompt_builder_code_snippet_preservation():
    code_snippet = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)"
    fp = FuzzyQueryProfile(
        code_focus=0.95,
        technical_depth=0.70,
    )
    prompt = dynamic_prompt_builder.build(
        message="Explain this recursive function",
        fuzzy_profile=fp,
        is_code=True,
        code_text=code_snippet,
    )

    assert code_snippet in prompt.messages[1]["content"]
    assert "Ground your explanation in code" in prompt.system_instruction


def test_prompt_builder_history_preservation():
    history = [
        {"role": "user", "content": "What is overfitting?"},
        {"role": "assistant", "content": "Overfitting happens when a model learns noise."},
    ]
    prompt = dynamic_prompt_builder.build(
        message="Why does it happen?",
        fuzzy_profile=FuzzyQueryProfile(),
        history=history,
    )

    assert len(prompt.messages) == 4  # system + user(1) + assistant(1) + user(2)
    assert prompt.messages[1]["content"] == "What is overfitting?"
    assert prompt.messages[2]["content"] == "Overfitting happens when a model learns noise."
    assert prompt.messages[3]["content"] == "Why does it happen?"


def test_streaming_and_non_streaming_prompt_formatting_consistency():
    """Verify that streaming and non-streaming modes construct identical canonical Gemma prompts."""
    from backend.app.services.chat_engine import format_messages_for_gemma

    history = [
        {"role": "user", "content": "What is self-attention?"},
        {"role": "assistant", "content": "Self-attention computes dynamic weights between query and key vectors."},
    ]
    rag_context = "[S1] Attention is all you need (Vaswani et al., 2017): Multi-head attention projects queries, keys, and values."

    built_prompt = dynamic_prompt_builder.build(
        message="How does multi-head attention extend this?",
        fuzzy_profile=FuzzyQueryProfile(technical_depth=0.8, aiml_relevance=0.9),
        history=history,
        rag_grounding=rag_context,
    )

    # 1. Non-streaming formatted messages (as used in generate())
    non_streaming_formatted = format_messages_for_gemma(built_prompt.messages)

    # 2. Streaming formatted messages (as used in generate_stream())
    streaming_formatted = format_messages_for_gemma(built_prompt.messages)

    # Both execution paths MUST receive identical formatted messages
    assert non_streaming_formatted == streaming_formatted

    # Verify no raw 'system' turn remains in formatted messages (Gemma requirement)
    roles = [m["role"] for m in non_streaming_formatted]
    assert "system" not in roles
    assert roles[0] == "user"

    # Verify that system instructions and RAG grounding are preserved in the first user turn
    first_user_content = non_streaming_formatted[0]["content"]
    assert "ChatLab" in first_user_content
    assert rag_context in first_user_content
    assert "What is self-attention?" in first_user_content

    # Verify that the subsequent turns and final user message are completely preserved
    assert non_streaming_formatted[1]["role"] == "assistant"
    assert non_streaming_formatted[1]["content"] == "Self-attention computes dynamic weights between query and key vectors."
    assert non_streaming_formatted[2]["role"] == "user"
    assert non_streaming_formatted[2]["content"] == "How does multi-head attention extend this?"

