from backend.app.services.intent_classifier import intent_classifier


def test_intent_classifier_loaded():
    assert intent_classifier.is_loaded is True
    assert intent_classifier.pipeline is not None


def test_intent_classifier_prediction_greeting():
    prediction = intent_classifier.predict("Hello there! Good morning.")
    assert prediction.intent == "greeting"
    assert prediction.confidence > 0.3
    assert 0.0 <= prediction.confidence <= 1.0


def test_intent_classifier_prediction_code_explanation():
    prediction = intent_classifier.predict("Explain this Python scikit-learn code for me")
    assert prediction.intent == "code_explanation"
    assert prediction.confidence > 0.3


def test_intent_classifier_prediction_unknown_or_low():
    prediction = intent_classifier.predict("Random gibberish xyz123 flimflam")
    assert prediction.intent in ["unknown", "capabilities", "help"] or prediction.confidence < 0.6

