from backend.app.services.semantic_similarity import semantic_similarity_engine


def test_similarity_engine_initialized():
    assert semantic_similarity_engine.is_initialized is True
    assert len(semantic_similarity_engine.patterns) > 0


def test_similarity_calculate_candidates():
    query = "explain how linear regression works"
    candidates = [
        "what is linear regression",
        "tell me a funny joke",
        "what is the weather today",
    ]
    score, best_match = semantic_similarity_engine.calculate_similarity(query, candidates)
    assert best_match == "what is linear regression"
    assert score > 0.4


def test_similarity_find_best_match():
    query = "start a machine learning quiz for me"
    match = semantic_similarity_engine.find_best_match(query)
    assert match.matched_intent in ["start_quiz", "quiz_topic"]
    assert match.score > 0.4
    assert len(match.matched_pattern) > 0

