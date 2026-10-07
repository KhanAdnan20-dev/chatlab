import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import joblib
from sklearn.pipeline import Pipeline
from backend.app.config import settings
from backend.app.services.preprocessing import preprocess


@dataclass
class IntentPrediction:
    """Encapsulates intent classification output and confidence."""

    intent: str
    confidence: float
    probabilities: dict[str, float]


class IntentClassifier:
    """
    Supervised Machine Learning Intent Classifier for Layer 1.
    Loads a serialized scikit-learn Pipeline (TF-IDF + Classifier) and computes
    actual calibrated probabilities for inference.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = Path(model_path or settings.INTENT_MODEL_PATH)
        self.pipeline: Optional[Pipeline] = None
        self._is_loaded = False
        self.load_model()

    @property
    def is_loaded(self) -> bool:
        """Returns whether the ML pipeline is loaded and ready for predictions."""
        return self._is_loaded

    def load_model(self) -> bool:
        """Loads the trained ML pipeline from disk artifact."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                self._is_loaded = True
                return True
            except Exception as e:
                print(f"[IntentClassifier] Error loading model from {self.model_path}: {e}")
                self.pipeline = None
                self._is_loaded = False
                return False
        return False

    def predict(self, text: str) -> IntentPrediction:
        """
        Classifies user input text into one of the trained intent classes
        and returns the prediction alongside true model probabilities.
        """
        if not self._is_loaded or self.pipeline is None:
            # Fallback when model is not yet trained/loaded
            return IntentPrediction(
                intent="unknown",
                confidence=0.0,
                probabilities={"unknown": 1.0},
            )

        preprocessed = preprocess(text)
        query = preprocessed.processed_text or preprocessed.normalized_text or text

        # Obtain class probabilities from the trained pipeline
        probabilities_array = self.pipeline.predict_proba([query])[0]
        classes = self.pipeline.classes_

        prob_dict = {
            cls_name: float(prob)
            for cls_name, prob in zip(classes, probabilities_array)
        }

        # Find class with highest probability
        best_intent = max(prob_dict, key=prob_dict.get)
        best_confidence = prob_dict[best_intent]

        return IntentPrediction(
            intent=best_intent,
            confidence=round(best_confidence, 4),
            probabilities=prob_dict,
        )


# Global singleton instance for efficient reuse across API requests
intent_classifier = IntentClassifier()
