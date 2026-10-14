"""
Grounded AI/ML Knowledge Retrieval Engine (RAG).

Provides dense semantic vector retrieval over verified textbooks and seminal papers
(Bishop PRML, Goodfellow Deep Learning, Vaswani et al., He et al., etc.).
Integrates directly with the pre-warmed MiniLM embedding model in memory,
delivering sub-millisecond retrieval with exact mathematical formulas and source attribution.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from backend.app.config import settings
from backend.app.services.semantic_similarity import semantic_similarity_engine

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """Represents a grounded chunk of verified AI/ML knowledge."""
    id: str
    concept_id: str
    title: str
    source: str
    section: str
    citation: str
    content: str
    formulas: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    similarity_score: float = 0.0


class RAGEngine:
    """
    In-memory vector retrieval engine for canonical AI/ML literature.
    """

    def __init__(self, corpus_path: Optional[Path] = None) -> None:
        self.corpus_path = corpus_path or (settings.BASE_DIR / "data" / "knowledge_corpus.json")
        self.items: List[KnowledgeItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self._is_initialized = False

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self) -> None:
        """Loads corpus JSON and encodes dense embeddings with TF-IDF fallback."""
        if not self.corpus_path.exists():
            logger.warning("[RAGEngine] Corpus file not found at %s", self.corpus_path)
            return

        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.items = [
                KnowledgeItem(
                    id=entry["id"],
                    concept_id=entry.get("concept_id", ""),
                    title=entry.get("title", ""),
                    source=entry.get("source", ""),
                    section=entry.get("section", ""),
                    citation=entry.get("citation", ""),
                    content=entry.get("content", ""),
                    formulas=entry.get("formulas", []),
                    keywords=entry.get("keywords", []),
                )
                for entry in data
            ]

            if not self.items:
                return

            texts_to_embed = [
                f"{item.title}. {item.content} {' '.join(item.keywords)}"
                for item in self.items
            ]

            # Index with TF-IDF as guaranteed baseline
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts_to_embed)

            dense_model = getattr(semantic_similarity_engine, "dense_model", None)
            if dense_model is not None:
                raw_embeds = dense_model.encode(
                    texts_to_embed,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self.embeddings = raw_embeds.astype(np.float32)
                self._is_initialized = True
                logger.info("[RAGEngine] Indexed %d knowledge chunks using dense embeddings.", len(self.items))
            else:
                self._is_initialized = True
                logger.info("[RAGEngine] Indexed %d knowledge chunks using TF-IDF baseline.", len(self.items))
        except Exception as exc:
            logger.error("[RAGEngine] Failed to load/index knowledge corpus: %s", exc)

    def retrieve(
        self,
        query: str,
        top_k: int = 2,
        threshold: float = 0.20,
    ) -> List[KnowledgeItem]:
        """
        Retrieves top-k relevant knowledge items matching query via cosine similarity.
        """
        if not self.items:
            self._load_and_index()
            if not self.items:
                return []

        # If dense model is now ready but embeddings were not yet generated, generate them
        dense_model = getattr(semantic_similarity_engine, "dense_model", None)
        if self.embeddings is None and dense_model is not None:
            texts_to_embed = [
                f"{item.title}. {item.content} {' '.join(item.keywords)}"
                for item in self.items
            ]
            self.embeddings = dense_model.encode(
                texts_to_embed,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

        try:
            if self.embeddings is not None and dense_model is not None:
                query_vec = dense_model.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0].astype(np.float32)

                scores = np.dot(self.embeddings, query_vec)
            elif self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                from sklearn.metrics.pairwise import cosine_similarity
                q_vec = self.tfidf_vectorizer.transform([query])
                scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]
            else:
                return []

            best_indices = np.argsort(scores)[::-1]
            matches: List[KnowledgeItem] = []

            for idx in best_indices[:top_k]:
                score = float(scores[idx])
                if score >= threshold:
                    matched_item = self.items[idx]
                    matched_item.similarity_score = round(score, 4)
                    matches.append(matched_item)

            return matches
        except Exception as exc:
            logger.error("[RAGEngine] Retrieval error: %s", exc)
            return []
        except Exception as exc:
            logger.error("[RAGEngine] Retrieval error: %s", exc)
            return []

    def format_grounding_prompt(self, items: List[KnowledgeItem]) -> str:
        """Formats retrieved items into system prompt grounding instructions."""
        if not items:
            return ""

        chunks_formatted = []
        for it in items:
            formulas_str = ("\nKey Mathematical Formulation(s):\n" + "\n".join(f"$$ {f} $$" for f in it.formulas)) if it.formulas else ""
            chunks_formatted.append(
                f"SOURCE: {it.source} ({it.section})\n"
                f"CITATION TAG: {it.citation}\n"
                f"CANONICAL PASSAGE:\n{it.content}{formulas_str}"
            )

        grounding_block = (
            "\n\n[VERIFIED AI/ML KNOWLEDGE GROUNDING]\n"
            "Reference the following authoritative literature in your explanation. "
            "Cite the corresponding citation tags (e.g. [Goodfellow-DL-§6.5]) and utilize the canonical mathematical formulas where appropriate:\n\n"
            + "\n\n---\n\n".join(chunks_formatted)
            + "\n[END KNOWLEDGE GROUNDING]\n"
        )
        return grounding_block


# Singleton instance
rag_engine = RAGEngine()
