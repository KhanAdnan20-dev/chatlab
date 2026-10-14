from backend.app.services.response_postprocessor import postprocess


def test_postprocess_normal():
    raw = "  Hello ChatLab!   \n\n\n  "
    res = postprocess(raw)
    assert res == "Hello ChatLab!"


def test_postprocess_empty():
    assert "Layer 2" in postprocess("")
    assert "Layer 2" in postprocess("   ")


def test_postprocess_truncation():
    long_text = "a" * 3000
    res = postprocess(long_text, max_chars=100)
    assert len(res) <= 105
    assert res.endswith("...")


def test_postprocess_unicode_repair():
    corrupt = "Machine learning\ufffdis a method where computers learn."
    res = postprocess(corrupt)
    assert "\ufffd" not in res
    assert " — " in res


def test_postprocess_markdown_balance():
    # Unclosed code fence
    unclosed_code = "Here is the code:\n```python\nprint('hello')"
    res_code = postprocess(unclosed_code)
    assert res_code.endswith("```")

    # Unclosed bold tag
    unclosed_bold = "This is **important concept that should close."
    res_bold = postprocess(unclosed_bold)
    assert res_bold.endswith("**") or res_bold.endswith(".**")


def test_postprocess_sentence_completion():
    truncated = "Supervised learning uses labeled data (e.g., house prices and features"
    res = postprocess(truncated)
    assert res.endswith(").") or res.endswith(".")

