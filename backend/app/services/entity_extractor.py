import json
import re
from pathlib import Path
from typing import Optional
from backend.app.config import settings
from backend.app.schemas import EntityItem

try:
    import nltk
    from nltk import ne_chunk, pos_tag, word_tokenize
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


CALENDAR_DAYS = {
    "monday": "DATE",
    "tuesday": "DATE",
    "wednesday": "DATE",
    "thursday": "DATE",
    "friday": "DATE",
    "saturday": "DATE",
    "sunday": "DATE",
    "today": "DATE",
    "tomorrow": "DATE",
    "yesterday": "DATE",
}


class EntityExtractor:
    """
    Modular Entity Extraction engine combining NLTK statistical POS/NER
    with AIML domain canonical concepts and alias resolution.
    """

    def __init__(self, topics_path: Optional[Path] = None) -> None:
        self.topics_path = Path(topics_path or settings.TOPICS_DATA_PATH)
        self.alias_to_canonical: dict[str, str] = {}
        self.load_aiml_topics()

    def load_aiml_topics(self) -> None:
        """Loads canonical AIML concepts and aliases from data/aiml_topics.json."""
        # Initialize with calendar days
        for day, label in CALENDAR_DAYS.items():
            self.alias_to_canonical[day] = f"DATE:{day}"

        if not self.topics_path.exists():
            return

        try:
            with open(self.topics_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            topics = data.get("topics", {})
            for canonical, aliases in topics.items():
                for alias in aliases:
                    self.alias_to_canonical[alias.strip().lower()] = canonical
        except Exception as e:
            print(f"[EntityExtractor] Warning loading topics: {e}")

    def extract_entities(self, text: str) -> list[EntityItem]:
        """
        Extracts named entities and canonical AIML concepts from raw text.
        Returns a deduplicated list of EntityItem objects with character spans.
        """
        if not text or not text.strip():
            return []

        entities: list[EntityItem] = []
        occupied_spans: list[tuple[int, int]] = []

        def is_overlapping(start: int, end: int) -> bool:
            for s, e in occupied_spans:
                if max(start, s) < min(end, e):
                    return True
            return False

        # 1. Match AIML concept dictionary and aliases (longest match first)
        sorted_aliases = sorted(
            self.alias_to_canonical.keys(), key=lambda k: len(k), reverse=True
        )

        lower_text = text.lower()
        for alias in sorted_aliases:
            canonical_tag = self.alias_to_canonical[alias]
            pattern = rf"\b{re.escape(alias)}\b"
            for match in re.finditer(pattern, lower_text):
                start, end = match.span()
                if not is_overlapping(start, end):
                    surface_text = text[start:end]
                    label = (
                        "DATE"
                        if canonical_tag.startswith("DATE:")
                        else f"CONCEPT:{canonical_tag}"
                    )
                    entities.append(
                        EntityItem(
                            text=surface_text,
                            label=label,
                            start=start,
                            end=end,
                        )
                    )
                    occupied_spans.append((start, end))

        # 2. Statistical NER for general entities (PERSON, ORGANIZATION, GPE)
        if _NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                tree = ne_chunk(pos_tags)

                for subtree in tree:
                    if hasattr(subtree, "label"):
                        chunk_label = subtree.label()
                        chunk_text = " ".join([leaf[0] for leaf in subtree.leaves()])

                        match = re.search(rf"\b{re.escape(chunk_text)}\b", text)
                        if match:
                            start, end = match.span()
                            if not is_overlapping(start, end):
                                entities.append(
                                    EntityItem(
                                        text=chunk_text,
                                        label=chunk_label,
                                        start=start,
                                        end=end,
                                    )
                                )
                                occupied_spans.append((start, end))
            except Exception:
                pass

        entities.sort(key=lambda e: (e.start if e.start is not None else 0))
        return entities


# Singleton instance reused across requests
entity_extractor = EntityExtractor()
