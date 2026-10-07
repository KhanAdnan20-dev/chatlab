from backend.app.services.entity_extractor import entity_extractor


def test_extract_programming_and_date_entities():
    text = "I have a Python exam on Monday."
    entities = entity_extractor.extract_entities(text)
    labels = {e.label for e in entities}
    texts = {e.text.lower() for e in entities}

    assert "python" in texts
    assert "monday" in texts
    assert "CONCEPT:python" in labels or "DATE" in labels
    assert "DATE" in labels


def test_extract_concept_entity():
    text = "Teach me machine learning and natural language processing."
    entities = entity_extractor.extract_entities(text)
    texts = [e.text.lower() for e in entities]
    labels = [e.label for e in entities]

    assert "machine learning" in texts
    assert "natural language processing" in texts
    assert "CONCEPT:machine_learning" in labels
    assert "CONCEPT:natural_language_processing" in labels


def test_empty_entity_extraction():
    entities = entity_extractor.extract_entities("")
    assert entities == []

