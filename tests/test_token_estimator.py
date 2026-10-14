"""Tests for NLP Response Demand Analyzer and Dynamic Token Budget Estimator."""
import pytest
from backend.app.schemas import EntityItem, FuzzyQueryProfile
from backend.app.services.token_estimator import (
    RESPONSE_TYPE_CODE_IMPLEMENTATION,
    RESPONSE_TYPE_CONVERSATIONAL,
    RESPONSE_TYPE_IN_DEPTH_EXPLANATION,
    RESPONSE_TYPE_STANDARD_CONCEPT,
    RESPONSE_TYPE_STEP_BY_STEP_DERIVATION,
    RESPONSE_TYPE_STRUCTURED_COMPARISON,
    RESPONSE_TYPE_ULTRA_CONCISE,
    token_budget_estimator,
)


def test_ultra_concise_response_demand():
    queries = [
        "What is overfitting in one sentence?",
        "Define deep learning briefly",
        "Give me a short answer on what a loss function is",
        "TL;DR of support vector machines",
    ]
    for q in queries:
        decision = token_budget_estimator.estimate(q)
        assert decision.response_type == RESPONSE_TYPE_ULTRA_CONCISE
        assert 48 <= decision.token_budget <= 96
        assert decision.demand_score <= 0.20


def test_conversational_response_demand():
    decision_greeting = token_budget_estimator.estimate("Hello there!", intent="greeting")
    assert decision_greeting.response_type == RESPONSE_TYPE_CONVERSATIONAL
    assert 48 <= decision_greeting.token_budget <= 80

    decision_id = token_budget_estimator.estimate("Who are you?", intent="identity")
    assert decision_id.response_type == RESPONSE_TYPE_CONVERSATIONAL
    assert decision_id.token_budget <= 80


def test_code_implementation_response_demand():
    decision_text = token_budget_estimator.estimate("Write Python code to implement linear regression from scratch")
    assert decision_text.response_type == RESPONSE_TYPE_CODE_IMPLEMENTATION
    assert decision_text.token_budget >= 384

    # Direct code snippet detection
    decision_code = token_budget_estimator.estimate(
        "def forward(x): return x",
        is_code=True,
    )
    assert decision_code.response_type == RESPONSE_TYPE_CODE_IMPLEMENTATION
    assert decision_code.token_budget >= 384


def test_step_by_step_derivation_demand():
    decision = token_budget_estimator.estimate(
        "Derive the gradient of binary cross-entropy loss step by step",
        fuzzy_profile=FuzzyQueryProfile(mathematical_detail=0.85, explanation_depth=0.80),
    )
    assert decision.response_type == RESPONSE_TYPE_STEP_BY_STEP_DERIVATION
    assert decision.token_budget >= 320


def test_structured_comparison_demand():
    decision = token_budget_estimator.estimate(
        "Compare bagging versus boosting with pros and cons",
    )
    assert decision.response_type == RESPONSE_TYPE_STRUCTURED_COMPARISON
    assert 384 <= decision.token_budget <= 768


def test_in_depth_explanation_demand():
    decision = token_budget_estimator.estimate(
        "Explain the transformer architecture in detail with complete self-attention mechanisms",
        fuzzy_profile=FuzzyQueryProfile(technical_depth=0.85, explanation_depth=0.85),
    )
    assert decision.response_type == RESPONSE_TYPE_IN_DEPTH_EXPLANATION
    assert decision.token_budget >= 416
    assert decision.demand_score >= 0.65


def test_standard_concept_demand():
    decision = token_budget_estimator.estimate("What is machine learning?")
    assert decision.response_type == RESPONSE_TYPE_STANDARD_CONCEPT
    assert 256 <= decision.token_budget <= 640


def test_multi_entity_multiplier():
    entities_single = [EntityItem(text="CNN", label="TECH_TERM")]
    entities_multi = [
        EntityItem(text="CNN", label="TECH_TERM"),
        EntityItem(text="RNN", label="TECH_TERM"),
        EntityItem(text="Transformer", label="TECH_TERM"),
    ]

    query = "Explain how CNN and RNN and Transformer handle sequence data"
    dec_single = token_budget_estimator.estimate(query, entities=entities_single)
    dec_multi = token_budget_estimator.estimate(query, entities=entities_multi)

    # Multi-entity query should receive equal or higher token budget
    assert dec_multi.token_budget >= dec_single.token_budget


def test_depth_mode_budget_allocation():
    query = "What is backpropagation?"

    dec_easy = token_budget_estimator.estimate(query, depth_mode="easy")
    assert dec_easy.token_budget == 384
    assert dec_easy.demand_score == 0.35

    dec_med = token_budget_estimator.estimate(query, depth_mode="medium")
    assert dec_med.token_budget == 640
    assert dec_med.demand_score == 0.65

    dec_hard = token_budget_estimator.estimate(query, depth_mode="hard")
    assert dec_hard.token_budget == 1024
    assert dec_hard.demand_score == 1.0

    dec_ultra = token_budget_estimator.estimate(query, depth_mode="ultra")
    assert dec_ultra.token_budget == 2048
    assert dec_ultra.demand_score == 1.0

    dec_auto = token_budget_estimator.estimate(query, depth_mode="auto")
    assert dec_auto.token_budget > 0

