"""
ChatLab Grounded RAG Retrieval Service.

Integrates the LangChain FAISS index and local all-MiniLM-L6-v2 embedding model
into the conversational pipeline. Provides intelligent retrieval gating, lazy
singleton index loading, graceful degradation, and citation metadata construction.
"""

import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import settings
from backend.app.rag.pipeline import (
    IndexValidationError,
    SearchResult,
    load_and_validate_index,
    search_index,
)
from backend.app.schemas import CitationItem, FuzzyQueryProfile

logger = logging.getLogger(__name__)

# Regular expression pattern to detect simple arithmetic calculations (e.g. "2+2", "what is 2 + 2?", "5 * 10")
ARITHMETIC_PATTERN = re.compile(
    r"^\s*(?:what\s+is\s+)?-?\d+(?:\.\d+)?\s*[\+\-\*\/\^%]\s*-?\d+(?:\.\d+)?\s*\??\s*$",
    re.IGNORECASE,
)

# Common conversational greetings, farewells, gratitude, and small talk phrases
CASUAL_PHRASES = {
    "hi",
    "hello",
    "hey",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "thx",
    "bye",
    "goodbye",
    "see you",
    "who are you",
    "what are you",
    "what can you do",
    "how are you",
    "how's it going",
    "help",
}

# Explicit user triggers asking for textbook, literature, or notes retrieval
EXPLICIT_RETRIEVAL_KEYWORDS = [
    "according to my notes",
    "according to notes",
    "in my notes",
    "available notes",
    "in the textbook",
    "in the book",
    "according to the book",
    "in the literature",
    "cite sources",
    "with citations",
    "reference textbook",
    "from the corpus",
    "from the pdf",
]

# Keywords strongly indicative of technical AI/ML questions
TECHNICAL_KEYWORDS = [
    "attention",
    "transformer",
    "backpropagation",
    "backprop",
    "gradient",
    "loss",
    "convolution",
    "regularization",
    "overfitting",
    "underfitting",
    "dropout",
    "normalization",
    "batchnorm",
    "layernorm",
    "eigenvector",
    "eigenvalue",
    "matrix",
    "chain rule",
    "optimizer",
    "adam",
    "sgd",
    "activation",
    "relu",
    "softmax",
    "sigmoid",
    "embedding",
    "latent",
    "encoder",
    "decoder",
    "autoencoder",
    "gan",
    "diffusion",
    "bayes",
    "markov",
    "perplexity",
    "cross-entropy",
    "mle",
    "map",
    "kl divergence",
    "support vector",
    "svm",
    "decision tree",
    "random forest",
    "xgboost",
    "clustering",
    "kmeans",
    "pca",
    "q-learning",
    "reinforcement learning",
    "rnn",
    "lstm",
    "gru",
    "bert",
    "gpt",
    "llm",
    "tokenizer",
    "self-attention",
    "multi-head",
]


class RAGRetrievalService:
    """
    Dedicated singleton RAG Retrieval Service.
    Loads vector store once per application lifecycle and delivers grounded context & citations.
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        model_path: Optional[Path] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> None:
        self.index_dir = index_dir or settings.RAG_INDEX_DIR
        self.model_path = model_path or settings.EMBEDDING_MODEL_PATH
        self.top_k = top_k or settings.RAG_TOP_K
        self.min_score = min_score if min_score is not None else settings.RAG_MIN_SCORE

        self._vector_store = None
        self._manifest: Optional[Dict[str, Any]] = None
        self._load_error: Optional[str] = None
        self._is_loaded = False
        self._lock = threading.Lock()
        self._mock_results: Optional[List[SearchResult]] = None

    @property
    def is_available(self) -> bool:
        """Indicates whether vector store is resident and ready for queries."""
        return self._vector_store is not None or self._mock_results is not None

    def ensure_loaded(self) -> bool:
        """
        Thread-safe lazy initialization of FAISS index and embedding model.
        Returns True if index is loaded and ready, False otherwise.
        """
        if self.is_available:
            return True

        with self._lock:
            if self.is_available:
                return True

            if not settings.RAG_ENABLED:
                logger.info("[RAGService] RAG retrieval is disabled via configuration.")
                return False

            try:
                logger.info(
                    "[RAGService] Loading FAISS index from %s using embedding model %s...",
                    self.index_dir,
                    self.model_path,
                )
                vs, manifest = load_and_validate_index(
                    index_dir=self.index_dir,
                    model_path=self.model_path,
                )
                self._vector_store = vs
                self._manifest = manifest
                self._load_error = None
                self._is_loaded = True
                logger.info(
                    "[RAGService] FAISS vector store ready (%d vectors, dim=%d).",
                    vs.index.ntotal,
                    manifest.get("vector_dimension", 384),
                )
                return True
            except IndexValidationError as ive:
                self._load_error = f"Index validation failed: {ive}"
                logger.warning("[RAGService] %s; degrading gracefully without RAG.", self._load_error)
                return False
            except FileNotFoundError as fnf:
                self._load_error = f"Index artifacts missing: {fnf}"
                logger.warning("[RAGService] %s; degrading gracefully without RAG.", self._load_error)
                return False
            except Exception as exc:
                self._load_error = f"Unexpected error loading FAISS index: {exc}"
                logger.error("[RAGService] %s; degrading gracefully without RAG.", self._load_error, exc_info=True)
                return False

    def should_retrieve(
        self,
        message: str,
        intent: Optional[str] = None,
        confidence: float = 0.0,
        fuzzy_profile: Optional[FuzzyQueryProfile] = None,
        entities: Optional[List[Any]] = None,
        is_code: bool = False,
    ) -> Tuple[bool, str]:
        """
        Decides whether RAG retrieval should be triggered for this conversational turn.
        Returns a tuple of (should_retrieve, decision_reason).
        """
        msg_clean = message.strip()
        msg_lower = msg_clean.lower()

        # 1. Skip simple arithmetic
        if ARITHMETIC_PATTERN.match(msg_clean):
            return False, "Simple arithmetic calculation; retrieval bypassed."

        # 2. Skip casual greetings, pleasantries, small talk
        if msg_lower in CASUAL_PHRASES or any(
            msg_lower == phrase for phrase in ["hi", "hello", "hey", "thanks", "thank you", "bye"]
        ):
            return False, "Conversational pleasantry/greeting; retrieval bypassed."

        if intent in ("greeting", "goodbye", "thanks", "identity", "capabilities", "chitchat"):
            # Check if there is an explicit request for notes embedded in the greeting
            if not any(k in msg_lower for k in EXPLICIT_RETRIEVAL_KEYWORDS):
                return False, f"Conversational intent '{intent}'; retrieval bypassed."

        # 3. Explicit retrieval requested by user (e.g., "according to my notes...")
        for keyword in EXPLICIT_RETRIEVAL_KEYWORDS:
            if keyword in msg_lower:
                return True, f"Explicit literature/notes retrieval trigger matched: '{keyword}'."

        # 4. Check for strong technical AI/ML keywords
        for tech_kw in TECHNICAL_KEYWORDS:
            # Word boundary check for short terms like "gan", "mle", "map", "pca"
            pattern = r"\b" + re.escape(tech_kw) + r"\b"
            if re.search(pattern, msg_lower):
                return True, f"Technical AI/ML terminology detected: '{tech_kw}'."

        # 5. Technical entities extracted by NLP layer
        if entities:
            tech_entities = [e for e in entities if getattr(e, "label", "") in ("TECH_TERM", "CONCEPT", "ALGORITHM")]
            if tech_entities:
                return True, f"Extracted technical entity: '{tech_entities[0].text}'."

        # 6. Mamdani Fuzzy Query Profile guidance
        if fuzzy_profile:
            # If domain relevance is high and technical depth is non-trivial
            if fuzzy_profile.aiml_relevance >= 0.60 and fuzzy_profile.technical_depth >= 0.40:
                return True, (
                    f"Fuzzy Profile triggered (aiml_relevance={fuzzy_profile.aiml_relevance:.2f}, "
                    f"technical_depth={fuzzy_profile.technical_depth:.2f})."
                )
            # If query is deemed out-of-domain
            if fuzzy_profile.aiml_relevance < 0.35:
                return False, f"Low AIML domain relevance ({fuzzy_profile.aiml_relevance:.2f}); retrieval bypassed."

        # 7. Check if classified intent is a technical concept
        if intent in (
            "machine_learning",
            "deep_learning",
            "neural_networks",
            "nlp_llms",
            "optimization",
            "computer_vision",
            "reinforcement_learning",
            "supervised_learning",
            "unsupervised_learning",
            "concept_explanation",
            "math_derivation",
        ):
            return True, f"Technical AIML intent '{intent}' classified."

        # Default fallback: general question without technical markers does not retrieve
        return False, "General question without technical AI/ML triggers; retrieval bypassed."

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[CitationItem], Optional[str]]:
        """
        Executes semantic retrieval against the local FAISS index.
        Returns:
            (citations, grounding_prompt_block)
        """
        if not query or not query.strip():
            return [], None

        effective_k = top_k or self.top_k

        # Check mock results first (used in unit test fixtures)
        if self._mock_results is not None:
            results = self._mock_results[:effective_k]
        else:
            if not self.ensure_loaded():
                logger.warning("[RAGService] Vector store unavailable; returning empty citations.")
                return [], None

            try:
                raw_results = search_index(
                    vector_store=self._vector_store,
                    query=query.strip(),
                    top_k=effective_k,
                )
                # Filter by minimum similarity score threshold
                results = [r for r in raw_results if r.score >= self.min_score]
            except Exception as exc:
                logger.error("[RAGService] Retrieval error for query '%s': %s", query, exc, exc_info=True)
                return [], None

        if not results:
            return [], None

        # Build stable citation items and grounding prompt
        citations: List[CitationItem] = []
        grounding_passages: List[str] = []

        for idx, res in enumerate(results, start=1):
            citation_code = f"[S{idx}]"
            page_str = f"p. {res.page_start}" if res.page_start == res.page_end else f"pp. {res.page_start}-{res.page_end}"

            citation_item = CitationItem(
                citation_code=citation_code,
                source_title=res.source_title,
                section=f"PDF {page_str}",
                content_snippet=res.snippet,
                formulas=[],
                file_name=res.file_name,
                page_start=res.page_start,
                page_end=res.page_end,
                subject=res.subject,
                score=res.score,
                source_url=res.source_url,
            )
            citations.append(citation_item)

            # Format passage for Gemma prompt
            passage_entry = (
                f"{citation_code} {res.source_title} ({res.file_name}, PDF {page_str}):\n"
                f"{res.full_text.strip()}"
            )
            grounding_passages.append(passage_entry)

        grounding_block = self._format_grounding_block(grounding_passages)
        return citations, grounding_block

    def _format_grounding_block(self, passages: List[str]) -> str:
        """Formats grounding instructions and passages with strict anti-hallucination rules."""
        passages_text = "\n\n---\n\n".join(passages)
        return (
            "\n\n[AUTHORITATIVE LITERATURE EVIDENCE]\n"
            "The following verified textbook passages were retrieved to support your answer:\n\n"
            f"{passages_text}\n\n"
            "EVIDENCE GROUNDING INSTRUCTIONS:\n"
            "- Answer the user's actual question directly, accurately, and naturally.\n"
            "- Use the retrieved textbook passages when they materially support the explanation.\n"
            "- Distinguish verified facts supported by the literature from general intuition.\n"
            "- Do not claim the books say something that the retrieved text does not support.\n"
            "- Do not invent quotations, formulas, citations, page numbers or source names.\n"
            "- Cite supporting passages using their labels ([S1], [S2], etc.) where appropriate.\n"
            "- When the evidence is insufficient or does not address the question, acknowledge the limitation rather than pretending the sources answer it.\n"
            "- Keep answers consistent with the specified explanation depth, examples, and mathematical detail.\n"
            "- Avoid dumping retrieved passages verbatim or repeating the user's question.\n"
            "- Do not treat similarity scores or rank orders as factual confidence or probabilities.\n"
            "[END LITERATURE EVIDENCE]\n"
        )

    def set_mock_results(self, results: Optional[List[SearchResult]]) -> None:
        """Testing utility: inject mock retrieval results to avoid GPU weight loading."""
        self._mock_results = results

    def reset_for_tests(self) -> None:
        """Testing utility: reset state."""
        self._mock_results = None


# Singleton instance
rag_retrieval_service = RAGRetrievalService()
