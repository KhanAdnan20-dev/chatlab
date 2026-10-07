from backend.app.services.preprocessing import (
    clean_punctuation,
    lemmatize_tokens,
    normalize_text,
    normalize_whitespace,
    preprocess,
    tokenize,
)


def test_normalize_whitespace():
    raw = "   Hello    World   from   ChatLab!   \n\t "
    assert normalize_whitespace(raw) == "Hello World from ChatLab!"


def test_normalize_text():
    raw = "  Héllò,  WÖRLD!  "
    normalized = normalize_text(raw)
    assert "hello" in normalized
    assert "world" in normalized
    assert normalized == "hello, world!"


def test_tokenize():
    text = "how do i learn python and machine-learning?"
    tokens = tokenize(text)
    assert "how" in tokens
    assert "learn" in tokens
    assert "python" in tokens
    assert "machine-learning" in tokens or "learning" in tokens


def test_lemmatize_tokens():
    tokens = ["studies", "cats", "running", "better"]
    lemmas = lemmatize_tokens(tokens)
    assert len(lemmas) == len(tokens)
    assert lemmas[0] in ["study", "studies"]


def test_preprocess_pipeline():
    text = "  Tell me   about your CAPABILITIES!  "
    result = preprocess(text)
    assert result.original_text == text
    assert "capabilities" in result.normalized_text
    assert len(result.tokens) > 0
    assert "capabilities" in result.processed_text or "capability" in result.processed_text
