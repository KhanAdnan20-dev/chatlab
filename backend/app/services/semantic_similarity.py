import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.app.config import settings
from backend.app.services.preprocessing import preprocess

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class SimilarityMatch:
    """Encapsulates semantic similarity matching output."""

    score: float
    matched_pattern: str
    matched_intent: str


class SemanticSimilarityEngine:
    """
    Computes semantic relatedness between user inputs and known dataset utterances.
    Primary: Local dense sentence embeddings (all-MiniLM-L6-v2).
    Fallback: Sub-word TF-IDF cosine vector space.
    Caches training pattern embeddings in memory to ensure fast sub-millisecond lookups.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
    ) -> None:
        self.data_path = Path(data_path or settings.INTENT_DATA_PATH)
        self.dense_model_path = Path(model_path or settings.EMBEDDING_MODEL_PATH)

        self.patterns: list[str] = []
        self.intents: list[str] = []
        self.pattern_embeddings: Optional[np.ndarray] = None

        # Fallback TF-IDF vectorizer
        self.fallback_vectorizer = TfidfVectorizer(
            ngram_range=(2, 5),
            analyzer="char_wb",
            sublinear_tf=True,
        )
        self.fallback_matrix = None

        self.dense_model = None
        self._is_initialized = False

        self._initialize_engine()

    @property
    def is_initialized(self) -> bool:
        """Returns True if training patterns have been indexed into vector space."""
        return self._is_initialized

    def _initialize_engine(self) -> None:
        """Loads dataset and indexes pattern embeddings."""
        if not self.data_path.exists():
            return

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_patterns: list[str] = []
            intents_list: list[str] = []

            for item in data.get("intents", []):
                intent_name = item.get("intent", "unknown")
                for pattern in item.get("patterns", []):
                    raw_patterns.append(pattern)
                    intents_list.append(intent_name)

            if not raw_patterns:
                return

            self.patterns = raw_patterns
            self.intents = intents_list

            # Try loading local dense sentence-transformer model first
            if _SENTENCE_TRANSFORMERS_AVAILABLE and self.dense_model_path.exists():
                try:
                    self.dense_model = SentenceTransformer(str(self.dense_model_path), device="cpu")
                    # Encode and normalize all patterns
                    embeddings = self.dense_model.encode(
                        self.patterns,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    self.pattern_embeddings = np.array(embeddings)
                    self._is_initialized = True
                    return
                except Exception as e:
                    print(f"[SemanticSimilarityEngine] Local MiniLM load warning: {e}")
                    self.dense_model = None

            # Fallback to TF-IDF vector space
            processed_patterns = [
                preprocess(p).processed_text or preprocess(p).normalized_text
                for p in raw_patterns
            ]
            self.fallback_matrix = self.fallback_vectorizer.fit_transform(processed_patterns)
            self._is_initialized = True
        except Exception as e:
            print(f"[SemanticSimilarityEngine] Initialization error: {e}")
            self._is_initialized = False

    def calculate_similarity(self, query: str, candidates: list[str]) -> tuple[float, str]:
        """
        Calculates similarity between a query and a custom list of candidate strings.
        Returns (highest_similarity_score, best_candidate_string).
        """
        if not candidates or not query:
            return 0.0, ""

        if self.dense_model is not None:
            try:
                q_emb = self.dense_model.encode([query], normalize_embeddings=True)
                c_emb = self.dense_model.encode(candidates, normalize_embeddings=True)
                sims = np.dot(c_emb, q_emb.T).flatten()
                best_idx = int(np.argmax(sims))
                return float(round(float(sims[best_idx]), 4)), candidates[best_idx]
            except Exception:
                pass

        # TF-IDF Fallback
        query_prep = preprocess(query).processed_text or query
        candidate_preps = [preprocess(c).processed_text or c for c in candidates]
        temp_vec = TfidfVectorizer(ngram_range=(2, 5), analyzer="char_wb", sublinear_tf=True)
        try:
            c_mat = temp_vec.fit_transform(candidate_preps)
            q_mat = temp_vec.transform([query_prep])
            sims = cosine_similarity(q_mat, c_mat).flatten()
            best_idx = int(np.argmax(sims))
            return float(round(float(sims[best_idx]), 4)), candidates[best_idx]
        except Exception:
            return 0.0, candidates[0] if candidates else ""

    def find_best_match(self, query: str) -> SimilarityMatch:
        """
        Searches the pre-indexed intent pattern space for the closest matching pattern
        and returns the corresponding similarity score and intent tag.
        """
        if not self._is_initialized or not self.patterns:
            return SimilarityMatch(score=0.0, matched_pattern="", matched_intent="unknown")

        # Dense model path
        if self.dense_model is not None and self.pattern_embeddings is not None:
            try:
                q_emb = self.dense_model.encode([query], normalize_embeddings=True)[0]
                similarities = np.dot(self.pattern_embeddings, q_emb)
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])
                return SimilarityMatch(
                    score=round(max(0.0, best_score), 4),
                    matched_pattern=self.patterns[best_idx],
                    matched_intent=self.intents[best_idx],
                )
            except Exception as e:
                print(f"[SemanticSimilarityEngine] Dense inference error: {e}")

        # TF-IDF Fallback path
        if self.fallback_matrix is not None:
            query_prep = preprocess(query).processed_text or preprocess(query).normalized_text or query
            query_vec = self.fallback_vectorizer.transform([query_prep])
            similarities = cosine_similarity(query_vec, self.fallback_matrix).flatten()
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            return SimilarityMatch(
                score=round(max(0.0, best_score), 4),
                matched_pattern=self.patterns[best_idx],
                matched_intent=self.intents[best_idx],
            )

        return SimilarityMatch(score=0.0, matched_pattern="", matched_intent="unknown")

    # Convenience alias
    compute_similarity = find_best_match


# Singleton instance reused across requests
semantic_similarity_engine = SemanticSimilarityEngine()
