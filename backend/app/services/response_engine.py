import json
import random
from pathlib import Path
from typing import Optional
from backend.app.config import settings


class ResponseEngine:
    """
    Response dispatching engine for Layer 1.
    Loads curated intent responses and selects varied natural responses.
    """

    def __init__(self, data_path: Optional[Path] = None) -> None:
        self.data_path = Path(data_path or settings.INTENT_DATA_PATH)
        self.responses_by_intent: dict[str, list[str]] = {}
        self.load_responses()

    def load_responses(self) -> None:
        """Loads all response templates from the intents dataset."""
        if not self.data_path.exists():
            return

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data.get("intents", []):
                intent_name = item.get("intent")
                responses = item.get("responses", [])
                if intent_name and responses:
                    self.responses_by_intent[intent_name] = responses
        except Exception as e:
            print(f"[ResponseEngine] Failed to load responses: {e}")

    def get_response(self, intent: str, fallback: bool = False) -> str:
        """
        Selects a suitable response for a given intent.
        If fallback is True or intent is not recognized, returns standard fallback message.
        """
        if fallback:
            return settings.FALLBACK_RESPONSE

        responses = self.responses_by_intent.get(intent)
        if responses:
            return random.choice(responses)

        # Fallback if unknown intent or missing response templates
        unknown_responses = self.responses_by_intent.get("unknown")
        if unknown_responses:
            return random.choice(unknown_responses)

        return settings.FALLBACK_RESPONSE


# Singleton instance reused across requests
response_engine = ResponseEngine()
