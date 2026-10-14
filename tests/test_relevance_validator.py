"""Tests for Response Relevance and Quality Validator."""
import pytest
from backend.app.services.relevance_validator import relevance_validator


def test_validator_detects_empty():
    res = relevance_validator.validate("What is machine learning?", "")
    assert not res.is_relevant
    assert res.should_retry is True


def test_validator_detects_system_leak():
    leaked_response = "You are ChatLab, an intelligent conversational AI assistant. Overfitting is..."
    res = relevance_validator.validate("What is overfitting?", leaked_response)
    assert not res.is_relevant
    assert res.should_retry is True


def test_validator_loocv_math_verification():
    # Valid answer containing 900
    valid_loocv = (
        "With 1,000 total samples and 100 samples reserved for test, 900 samples remain. "
        "In LOOCV, each remaining sample is left out once, resulting in 900 validation splits."
    )
    res_valid = relevance_validator.validate(
        "Calculate the number of LOOCV validation splits when 100 of 1,000 samples are reserved for testing.",
        valid_loocv,
    )
    assert res_valid.is_relevant is True

    # Invalid answer lacking 900
    invalid_loocv = "LOOCV creates 1000 folds because leave-one-out always equals sample count."
    res_invalid = relevance_validator.validate(
        "Calculate the number of LOOCV validation splits when 100 of 1,000 samples are reserved for testing.",
        invalid_loocv,
    )
    assert res_invalid.is_relevant is False
    assert "900" in res_invalid.reason


def test_validator_f1_score_arithmetic():
    # Valid answer for TP=72, FP=18, FN=10
    valid_f1 = (
        "Calculations:\n"
        "- Precision = 72 / (72 + 18) = 72 / 90 = 0.8000\n"
        "- Recall = 72 / (72 + 10) = 72 / 82 ≈ 0.8780\n"
        "- F1 = 2 * (0.8000 * 0.8780) / (0.8000 + 0.8780) ≈ 0.8372"
    )
    res_valid = relevance_validator.validate(
        "Calculate precision, recall, and F1-score for TP=72, FP=18, FN=10",
        valid_f1,
    )
    assert res_valid.is_relevant is True

    # Invalid answer
    invalid_f1 = "The precision is 0.50 and F1 is 0.60."
    res_invalid = relevance_validator.validate(
        "Calculate precision, recall, and F1-score for TP=72, FP=18, FN=10",
        invalid_f1,
    )
    assert res_invalid.is_relevant is False


def test_validator_code_relevance():
    code = "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    return -1"
    good_resp = "This binary_search function takes arr and target, initializing low and high pointers."
    res_good = relevance_validator.validate(
        "Explain this code",
        good_resp,
        is_code=True,
        code_text=code,
    )
    assert res_good.is_relevant is True

    bad_resp = "Linear regression is a statistical method for modeling linear relationships."
    res_bad = relevance_validator.validate(
        "Explain this code",
        bad_resp,
        is_code=True,
        code_text=code,
    )
    assert res_bad.is_relevant is False
