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
    )
    assert decision.selected_layer == "layer1"
    assert decision.activation_score <= 0.40
    assert decision.is_fallback is False
    assert 0.0 <= decision.activation_score <= 1.0


def test_fuzzy_router_low_confidence_complex_query():
    decision = fuzzy_router.route(
        confidence=0.15,
        similarity=0.20,
        complexity=0.85,
        entity_density=0.0,
        intent="unknown",
    )
    assert decision.selected_layer == "layer2_pending"
    assert decision.activation_score >= 0.50
    assert decision.is_fallback is True
    assert 0.0 <= decision.activation_score <= 1.0


def test_fuzzy_router_medium_confidence_query():
    decision = fuzzy_router.route(
        confidence=0.50,
        similarity=0.50,
        complexity=0.50,
        entity_density=0.25,
        intent="study_help",
    )
    assert 0.0 <= decision.activation_score <= 1.0
    assert len(decision.reason) > 0


def test_fuzzy_router_boundary_values():
    # All zeros
    decision_zero = fuzzy_router.route(0.0, 0.0, 0.0, 0.0, intent="unknown")
    assert 0.0 <= decision_zero.activation_score <= 1.0
    assert decision_zero.selected_layer == "layer2_pending"

    # All ones
    decision_one = fuzzy_router.route(1.0, 1.0, 1.0, 1.0, intent="greeting")
    assert 0.0 <= decision_one.activation_score <= 1.0


def test_compute_complexity_and_entity_density():
    raw = "What is the Pomodoro technique for studying?"
    prep = preprocess(raw)
    complexity = fuzzy_router.compute_query_complexity(raw, prep)
    assert 0.0 <= complexity <= 1.0
    assert complexity > 0.1

    density = fuzzy_router.compute_entity_density([], prep)
    assert density == 0.0
