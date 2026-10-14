import pytest
from backend.app.services.fuzzy_router import (
    FuzzyInputs,
    fuzzy_router,
    trapmf,
    trimf,
)
from backend.app.services.preprocessing import preprocess


def test_membership_functions_bounds():
    # Test trimf
    assert trimf(0.0, [0.2, 0.5, 0.8]) == 0.0
    assert trimf(0.5, [0.2, 0.5, 0.8]) == 1.0
    assert trimf(1.0, [0.2, 0.5, 0.8]) == 0.0
    assert 0.0 < trimf(0.35, [0.2, 0.5, 0.8]) < 1.0

    # Test trapmf
    assert trapmf(0.0, [0.0, 0.0, 0.2, 0.4]) == 1.0
    assert trapmf(0.1, [0.0, 0.0, 0.2, 0.4]) == 1.0
    assert trapmf(0.3, [0.0, 0.0, 0.2, 0.4]) == pytest.approx(0.5, abs=1e-5)
    assert trapmf(1.0, [0.6, 0.8, 1.0, 1.0]) == 1.0
    assert trapmf(0.0, [0.6, 0.8, 1.0, 1.0]) == 0.0


def test_fuzzy_router_high_confidence_simple_query():
    decision = fuzzy_router.route(
        confidence=0.92,
        similarity=0.95,
        complexity=0.15,
        entity_density=0.0,
        intent="greeting",
        text="hello",
    )
    assert decision.selected_layer in ["layer1", "layer2"]
    assert decision.fuzzy_profile is not None
    assert decision.fuzzy_profile.conversationalness >= 0.50
    assert 0.0 <= decision.activation_score <= 1.0


def test_fuzzy_router_low_confidence_complex_query():
    decision = fuzzy_router.route(
        confidence=0.15,
        similarity=0.20,
        complexity=0.85,
        entity_density=0.0,
        intent="unknown",
        text="teach me advanced deep neural network architectures",
    )
    assert decision.selected_layer in ["layer2", "layer2_pending"]
    assert decision.fuzzy_profile.technical_depth >= 0.40
    assert 0.0 <= decision.activation_score <= 1.0


def test_fuzzy_router_medium_confidence_query():
    decision = fuzzy_router.route(
        confidence=0.50,
        similarity=0.50,
        complexity=0.50,
        entity_density=0.25,
        intent="study_help",
        text="how to optimize loss function",
    )
    assert 0.0 <= decision.activation_score <= 1.0
    assert len(decision.reason) > 0
    assert decision.fuzzy_profile is not None


def test_fuzzy_router_boundary_values():
    # All zeros
    decision_zero = fuzzy_router.route(0.0, 0.0, 0.0, 0.0, intent="unknown")
    assert 0.0 <= decision_zero.activation_score <= 1.0
    assert decision_zero.selected_layer in ["layer2", "layer2_pending"]

    # All ones
    decision_one = fuzzy_router.route(1.0, 1.0, 1.0, 1.0, intent="greeting", text="hello")
    assert 0.0 <= decision_one.activation_score <= 1.0


def test_compute_complexity_and_entity_density():
    raw = "What is the Pomodoro technique for studying?"
    prep = preprocess(raw)
    complexity = fuzzy_router.compute_query_complexity(raw, prep)
    assert 0.0 <= complexity <= 1.0
    assert complexity > 0.1

    density = fuzzy_router.compute_entity_density([], prep)
    assert density == 0.0


def test_fuzzy_router_depth_mode_overrides():
    # Easy mode
    easy_dec = fuzzy_router.route(
        confidence=0.8,
        similarity=0.8,
        complexity=0.5,
        entity_density=0.2,
        intent="concept_explanation",
        text="What is gradient descent?",
        depth_mode="easy",
    )
    assert easy_dec.fuzzy_profile.technical_depth <= 0.35
    assert easy_dec.fuzzy_profile.beginner_friendliness >= 0.85
    assert easy_dec.fuzzy_profile.explanation_depth <= 0.45
    assert any("Easy" in r for r in easy_dec.fired_rules)

    # Medium mode
    med_dec = fuzzy_router.route(
        confidence=0.8,
        similarity=0.8,
        complexity=0.5,
        entity_density=0.2,
        intent="concept_explanation",
        text="What is gradient descent?",
        depth_mode="medium",
    )
    assert 0.55 <= med_dec.fuzzy_profile.technical_depth <= 0.75
    assert 0.55 <= med_dec.fuzzy_profile.explanation_depth <= 0.75
    assert any("Medium" in r for r in med_dec.fired_rules)

    # Hard mode
    hard_dec = fuzzy_router.route(
        confidence=0.8,
        similarity=0.8,
        complexity=0.5,
        entity_density=0.2,
        intent="concept_explanation",
        text="What is gradient descent?",
        depth_mode="hard",
    )
    assert hard_dec.fuzzy_profile.technical_depth >= 0.85
    assert hard_dec.fuzzy_profile.mathematical_detail >= 0.85
    assert hard_dec.fuzzy_profile.explanation_depth >= 0.85
    assert any("Hard" in r for r in hard_dec.fired_rules)

    # Ultra mode
    ultra_dec = fuzzy_router.route(
        confidence=0.8,
        similarity=0.8,
        complexity=0.5,
        entity_density=0.2,
        intent="concept_explanation",
        text="What is gradient descent?",
        depth_mode="ultra",
    )
    assert ultra_dec.fuzzy_profile.technical_depth == 1.0
    assert ultra_dec.fuzzy_profile.mathematical_detail >= 0.95
    assert ultra_dec.fuzzy_profile.explanation_depth == 1.0
    assert any("Ultra" in r for r in ultra_dec.fired_rules)


def test_fuzzy_router_auto_mode_nlp_keyword_matching():
    # Query with deep keywords in auto mode
    deep_dec = fuzzy_router.route(
        confidence=0.7,
        similarity=0.7,
        complexity=0.6,
        entity_density=0.3,
        intent="concept_explanation",
        text="Explain in-depth architecture and mathematical derivation of backpropagation step-by-step",
        depth_mode="auto",
    )
    assert deep_dec.fuzzy_profile.technical_depth >= 0.65
    assert deep_dec.fuzzy_profile.explanation_depth >= 0.65
    assert deep_dec.fuzzy_profile.mathematical_detail >= 0.65

