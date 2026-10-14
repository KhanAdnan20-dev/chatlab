#!/usr/bin/env python3
"""
ChatLab Phase 2C — Offline Retrieval Evaluation and Evidence Quality Benchmark.

Systematically evaluates retrieval quality across:
- Dense FAISS inner-product retrieval (baseline).
- Sparse BM25 retrieval.
- Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Outputs comprehensive metrics, failure case taxonomy, and query-level logs to:
data/rag/evaluation/results.json
"""

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings
from backend.app.rag.pipeline import (
    IndexValidationError,
    SearchResult,
    load_and_validate_index,
    search_index,
)

logger = logging.getLogger("evaluate_rag")


# ============================================================================
# BM25 Sparse Inverted Index Implementation
# ============================================================================

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())


class LocalBM25Index:
    """
    In-memory Okapi BM25 retriever for local RAG chunks.
    Preserves exact chunk metadata and page provenance.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_records: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.chunk_id_to_idx: Dict[str, int] = {}

    def build_from_documents(self, documents_path: Path):
        """Index all chunks from processed documents.jsonl."""
        logger.info("Building BM25 inverted index from %s...", documents_path)
        self.doc_records = []
        self.doc_lengths = []
        self.inverted_index = defaultdict(list)
        self.chunk_id_to_idx = {}

        with open(documents_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                rec = json.loads(line)
                self.doc_records.append(rec)
                self.chunk_id_to_idx[rec["chunk_id"]] = idx

                tokens = tokenize_text(rec["text"])
                doc_len = len(tokens)
                self.doc_lengths.append(doc_len)

                term_freqs = Counter(tokens)
                for term, freq in term_freqs.items():
                    self.inverted_index[term].append((idx, freq))

        N = len(self.doc_records)
        self.avg_doc_len = sum(self.doc_lengths) / max(N, 1)

        # Precompute Okapi BM25 IDF
        self.idf = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)
            # Standard Lucene/BM25Okapi IDF formula with +1.0 floor
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        logger.info(
            "BM25 index built: %d documents, %d unique terms, avgdl=%.1f",
            N,
            len(self.inverted_index),
            self.avg_doc_len,
        )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search BM25 inverted index for a query string."""
        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores: Dict[int, float] = defaultdict(float)
        query_counts = Counter(tokens)

        for term, q_weight in query_counts.items():
            if term not in self.inverted_index:
                continue
            term_idf = self.idf[term]
            for doc_idx, freq in self.inverted_index[term]:
                doc_len = self.doc_lengths[doc_idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
                )
                scores[doc_idx] += term_idf * (numerator / denominator) * q_weight

        if not scores:
            return []

        # Sort top documents by BM25 score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_idx, score) in enumerate(sorted_docs, start=1):
            doc = self.doc_records[doc_idx]
            text = doc["text"]
            snippet = text[:300] + ("..." if len(text) > 300 else "")
            results.append(
                SearchResult(
                    rank=rank,
                    score=round(score, 4),
                    source_title=doc["source_title"],
                    file_name=doc["file_name"],
                    page_start=doc["page_start"],
                    page_end=doc["page_end"],
                    subject=doc["subject"],
                    source_url=doc.get("source_url"),
                    snippet=snippet,
                    full_text=text,
                    chunk_id=doc["chunk_id"],
                )
            )

        return results


# ============================================================================
# Hybrid Reciprocal Rank Fusion (RRF)
# ============================================================================

def fuse_rankings_rrf(
    dense_results: List[SearchResult],
    sparse_results: List[SearchResult],
    top_k: int = 5,
    rrf_k: int = 60,
) -> List[SearchResult]:
    """
    Principled Reciprocal Rank Fusion (RRF).
    RRF score = sum( 1 / (rrf_k + rank_m) ) for m in {dense, sparse}.
    Combines scale-incompatible dense cosine/inner-product scores and BM25 scores.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, SearchResult] = {}

    for rank, res in enumerate(dense_results, start=1):
        rrf_scores[res.chunk_id] += 1.0 / (rrf_k + rank)
        chunk_map[res.chunk_id] = res

    for rank, res in enumerate(sparse_results, start=1):
        rrf_scores[res.chunk_id] += 1.0 / (rrf_k + rank)
        if res.chunk_id not in chunk_map:
            chunk_map[res.chunk_id] = res

    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fused_results = []
    for rank, (cid, score) in enumerate(sorted_chunks, start=1):
        orig = chunk_map[cid]
        fused_results.append(
            SearchResult(
                rank=rank,
                score=round(score, 5),
                source_title=orig.source_title,
                file_name=orig.file_name,
                page_start=orig.page_start,
                page_end=orig.page_end,
                subject=orig.subject,
                source_url=orig.source_url,
                snippet=orig.snippet,
                full_text=orig.full_text,
                chunk_id=orig.chunk_id,
            )
        )

    return fused_results


# ============================================================================
# Metric Evaluation Engine
# ============================================================================

@dataclass
class QueryEvaluationRecord:
    query_id: str
    query: str
    category: str
    is_supported: bool
    requires_human_annotation: bool
    expected_sources: List[str]
    expected_chunk_ids: List[str]
    expected_page_ranges: List[List[int]]
    retrieved_chunk_ids: List[str]
    retrieved_sources: List[str]
    retrieved_scores: List[float]
    retrieved_pages: List[List[int]]
    latency_ms: float
    # Computed metrics
    hit_at_1: Optional[bool]
    hit_at_3: Optional[bool]
    hit_at_5: Optional[bool]
    source_hit_at_1: Optional[bool]
    source_hit_at_3: Optional[bool]
    source_hit_at_5: Optional[bool]
    page_provenance_match: Optional[bool]
    reciprocal_rank: Optional[float]
    ndcg_at_5: Optional[float]
    is_gated_properly: Optional[bool]  # For unsupported: max score < 0.35
    error_class: Optional[str]


def compute_ndcg_at_k(relevant_retrieved_ranks: List[int], total_relevant: int, k: int = 5) -> float:
    """Compute nDCG@k with binary relevance labels."""
    if total_relevant == 0:
        return 0.0

    dcg = 0.0
    for rank in relevant_retrieved_ranks:
        if rank <= k:
            dcg += 1.0 / math.log2(rank + 1.0)

    idcg = 0.0
    for r in range(1, min(k, total_relevant) + 1):
        idcg += 1.0 / math.log2(r + 1.0)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_query_results(
    query_meta: Dict[str, Any],
    results: List[SearchResult],
    latency_ms: float,
    min_score_threshold: float = 0.35,
) -> QueryEvaluationRecord:
    """Evaluate retrieval results against ground-truth annotations for a single query."""
    qid = query_meta["query_id"]
    query = query_meta["query"]
    category = query_meta.get("category", "unknown")
    is_supported = query_meta.get("is_supported", True)
    requires_human = query_meta.get("requires_human_annotation", False)
    expected_sources = query_meta.get("expected_sources", [])
    expected_chunks = set(query_meta.get("expected_chunk_ids", []))
    expected_page_ranges = query_meta.get("expected_page_ranges", [])

    retrieved_cids = [r.chunk_id for r in results]
    retrieved_srcs = [r.file_name for r in results]
    retrieved_scores = [r.score for r in results]
    retrieved_pages = [[r.page_start, r.page_end] for r in results]

    # Unsupported queries: check gating score
    if not is_supported:
        max_score = max(retrieved_scores) if retrieved_scores else 0.0
        gated_properly = max_score < min_score_threshold
        error_class = None if gated_properly else "unsupported_false_positive"

        return QueryEvaluationRecord(
            query_id=qid,
            query=query,
            category=category,
            is_supported=False,
            requires_human_annotation=False,
            expected_sources=[],
            expected_chunk_ids=[],
            expected_page_ranges=[],
            retrieved_chunk_ids=retrieved_cids,
            retrieved_sources=retrieved_srcs,
            retrieved_scores=retrieved_scores,
            retrieved_pages=retrieved_pages,
            latency_ms=latency_ms,
            hit_at_1=None,
            hit_at_3=None,
            hit_at_5=None,
            source_hit_at_1=None,
            source_hit_at_3=None,
            source_hit_at_5=None,
            page_provenance_match=None,
            reciprocal_rank=None,
            ndcg_at_5=None,
            is_gated_properly=gated_properly,
            error_class=error_class,
        )

    # Ambiguous / unannotated queries: exclude from ground-truth metrics
    if requires_human:
        return QueryEvaluationRecord(
            query_id=qid,
            query=query,
            category=category,
            is_supported=True,
            requires_human_annotation=True,
            expected_sources=expected_sources,
            expected_chunk_ids=[],
            expected_page_ranges=[],
            retrieved_chunk_ids=retrieved_cids,
            retrieved_sources=retrieved_srcs,
            retrieved_scores=retrieved_scores,
            retrieved_pages=retrieved_pages,
            latency_ms=latency_ms,
            hit_at_1=None,
            hit_at_3=None,
            hit_at_5=None,
            source_hit_at_1=None,
            source_hit_at_3=None,
            source_hit_at_5=None,
            page_provenance_match=None,
            reciprocal_rank=None,
            ndcg_at_5=None,
            is_gated_properly=None,
            error_class="unannotated_needs_human_verification",
        )

    # Annotated supported queries: evaluate exact chunk matches and source/page matches
    relevant_ranks = []
    for rank, cid in enumerate(retrieved_cids, start=1):
        if cid in expected_chunks:
            relevant_ranks.append(rank)

    # Check if page matches within window (+/- 2 pages of expected)
    page_match = False
    for r in results:
        r_file = r.file_name
        r_start, r_end = r.page_start, r.page_end
        if r_file in expected_sources:
            for exp_start, exp_end in expected_page_ranges:
                if max(r_start, exp_start - 2) <= min(r_end, exp_end + 2):
                    page_match = True
                    break
        if page_match:
            break

    # If exact chunk is hit OR page matches in the expected source, count as hit
    hit_1 = (1 in relevant_ranks) or (
        len(retrieved_srcs) > 0
        and retrieved_srcs[0] in expected_sources
        and page_match
        and any(
            max(retrieved_pages[0][0], exp[0] - 2) <= min(retrieved_pages[0][1], exp[1] + 2)
            for exp in expected_page_ranges
        )
    )
    hit_3 = any(r <= 3 for r in relevant_ranks) or (
        any(s in expected_sources for s in retrieved_srcs[:3]) and page_match
    )
    hit_5 = any(r <= 5 for r in relevant_ranks) or (
        any(s in expected_sources for s in retrieved_srcs[:5]) and page_match
    )

    # Source-level matches
    src_hit_1 = len(retrieved_srcs) > 0 and retrieved_srcs[0] in expected_sources
    src_hit_3 = any(s in expected_sources for s in retrieved_srcs[:3])
    src_hit_5 = any(s in expected_sources for s in retrieved_srcs[:5])

    # Reciprocal rank
    rr = 0.0
    if relevant_ranks:
        rr = 1.0 / relevant_ranks[0]
    elif hit_5:
        # If page provenance matched at rank r
        for rank, (src, pg) in enumerate(zip(retrieved_srcs, retrieved_pages), start=1):
            if src in expected_sources and any(
                max(pg[0], exp[0] - 2) <= min(pg[1], exp[1] + 2) for exp in expected_page_ranges
            ):
                rr = 1.0 / rank
                break

    # nDCG@5
    effective_ranks = relevant_ranks if relevant_ranks else ([1] if hit_1 else ([3] if hit_3 else []))
    ndcg = compute_ndcg_at_k(effective_ranks, total_relevant=len(expected_chunks), k=5)

    # Error classification
    error_class = None
    if not hit_5:
        if not src_hit_5:
            error_class = "incorrect_source_match"
        elif any(p[0] < 15 for p in retrieved_pages[:3]):
            error_class = "toc_or_overview_dominance"
        elif category in ("formula_derivation", "concept_definition"):
            error_class = "formula_or_symbol_lookup_failure"
        elif category == "exact_terminology":
            error_class = "exact_technical_terminology_failure"
        else:
            error_class = "semantically_similar_non_answering"

    return QueryEvaluationRecord(
        query_id=qid,
        query=query,
        category=category,
        is_supported=True,
        requires_human_annotation=False,
        expected_sources=expected_sources,
        expected_chunk_ids=list(expected_chunks),
        expected_page_ranges=expected_page_ranges,
        retrieved_chunk_ids=retrieved_cids,
        retrieved_sources=retrieved_srcs,
        retrieved_scores=retrieved_scores,
        retrieved_pages=retrieved_pages,
        latency_ms=latency_ms,
        hit_at_1=hit_1,
        hit_at_3=hit_3,
        hit_at_5=hit_5,
        source_hit_at_1=src_hit_1,
        source_hit_at_3=src_hit_3,
        source_hit_at_5=src_hit_5,
        page_provenance_match=page_match,
        reciprocal_rank=rr,
        ndcg_at_5=round(ndcg, 4),
        is_gated_properly=None,
        error_class=error_class,
    )


def summarize_metrics(records: List[QueryEvaluationRecord]) -> Dict[str, Any]:
    """Calculate aggregate metrics over evaluation records."""
    annotated = [r for r in records if r.is_supported and not r.requires_human_annotation]
    unsupported = [r for r in records if not r.is_supported]
    unannotated = [r for r in records if r.requires_human_annotation]

    N = len(annotated)
    if N == 0:
        return {"error": "No annotated queries available"}

    recall_1 = sum(1 for r in annotated if r.hit_at_1) / N
    recall_3 = sum(1 for r in annotated if r.hit_at_3) / N
    recall_5 = sum(1 for r in annotated if r.hit_at_5) / N

    source_acc_1 = sum(1 for r in annotated if r.source_hit_at_1) / N
    source_acc_3 = sum(1 for r in annotated if r.source_hit_at_3) / N
    source_acc_5 = sum(1 for r in annotated if r.source_hit_at_5) / N

    page_provenance_acc = sum(1 for r in annotated if r.page_provenance_match) / N
    mrr = sum(r.reciprocal_rank or 0.0 for r in annotated) / N
    mean_ndcg = sum(r.ndcg_at_5 or 0.0 for r in annotated) / N

    # Unsupported queries gating accuracy
    unsupported_n = len(unsupported)
    gated_correct = (
        sum(1 for r in unsupported if r.is_gated_properly) if unsupported_n > 0 else 0
    )
    gating_accuracy = (gated_correct / unsupported_n) if unsupported_n > 0 else 1.0

    # Latencies
    all_latencies = [r.latency_ms for r in records]
    all_latencies.sort()
    mean_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    median_latency = all_latencies[len(all_latencies) // 2] if all_latencies else 0.0
    p95_latency = all_latencies[int(len(all_latencies) * 0.95)] if all_latencies else 0.0

    # Missed queries (Recall@5 = 0)
    missed_queries = [
        {
            "query_id": r.query_id,
            "query": r.query,
            "category": r.category,
            "expected_sources": r.expected_sources,
            "error_class": r.error_class,
            "top_retrieved": [
                {
                    "chunk_id": cid,
                    "file_name": src,
                    "score": score,
                    "pages": pg,
                }
                for cid, src, score, pg in zip(
                    r.retrieved_chunk_ids[:3],
                    r.retrieved_sources[:3],
                    r.retrieved_scores[:3],
                    r.retrieved_pages[:3],
                )
            ],
        }
        for r in annotated
        if not r.hit_at_5
    ]

    # Error classification counts
    error_counts = Counter(r.error_class for r in annotated if r.error_class)

    return {
        "dataset_summary": {
            "total_queries": len(records),
            "annotated_ground_truth_queries": N,
            "unannotated_queries": len(unannotated),
            "unsupported_negative_queries": unsupported_n,
        },
        "retrieval_metrics": {
            "recall_at_1": round(recall_1, 4),
            "recall_at_3": round(recall_3, 4),
            "recall_at_5": round(recall_5, 4),
            "mrr": round(mrr, 4),
            "ndcg_at_5": round(mean_ndcg, 4),
            "source_accuracy_at_1": round(source_acc_1, 4),
            "source_accuracy_at_3": round(source_acc_3, 4),
            "source_accuracy_at_5": round(source_acc_5, 4),
            "page_provenance_accuracy": round(page_provenance_acc, 4),
            "unsupported_gating_accuracy": round(gating_accuracy, 4),
            "zero_hit_count": len(missed_queries),
        },
        "latency_ms": {
            "mean": round(mean_latency, 2),
            "median": round(median_latency, 2),
            "p95": round(p95_latency, 2),
            "total_batch_time_s": round(sum(all_latencies) / 1000.0, 2),
        },
        "error_distribution": dict(error_counts),
        "missed_queries": missed_queries,
    }


# ============================================================================
# Main Evaluation Orchestration
# ============================================================================

def load_evaluation_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    queries = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def run_benchmark(
    mode: str,
    queries: List[Dict[str, Any]],
    vector_store: Any,
    bm25_index: Optional[LocalBM25Index] = None,
    top_k: int = 5,
    rrf_k: int = 60,
) -> Tuple[Dict[str, Any], List[QueryEvaluationRecord]]:
    """Execute evaluation for the specified mode (dense, bm25, or hybrid)."""
    records: List[QueryEvaluationRecord] = []

    for idx, qmeta in enumerate(queries, start=1):
        query_text = qmeta["query"]
        t0 = time.perf_counter()

        if mode == "dense":
            results = search_index(vector_store, query_text, top_k=top_k)
        elif mode == "bm25":
            if not bm25_index:
                raise ValueError("BM25 index not initialized for mode 'bm25'")
            results = bm25_index.search(query_text, top_k=top_k)
        elif mode == "hybrid":
            if not bm25_index:
                raise ValueError("BM25 index not initialized for mode 'hybrid'")
            # Retrieve candidates from both systems then fuse via RRF
            dense_res = search_index(vector_store, query_text, top_k=top_k * 3)
            sparse_res = bm25_index.search(query_text, top_k=top_k * 3)
            results = fuse_rankings_rrf(dense_res, sparse_res, top_k=top_k, rrf_k=rrf_k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        rec = evaluate_query_results(qmeta, results, latency_ms)
        records.append(rec)

    metrics = summarize_metrics(records)
    return metrics, records


def print_comparison_table(dense_m: Dict[str, Any], bm25_m: Dict[str, Any], hybrid_m: Dict[str, Any]):
    """Print ASCII comparison table across all three search modalities."""
    headers = ["Metric", "Dense (FAISS)", "BM25 (Sparse)", "Hybrid (RRF)", "Gain vs Dense"]
    rows = [
        ("Recall@1", "recall_at_1"),
        ("Recall@3", "recall_at_3"),
        ("Recall@5", "recall_at_5"),
        ("MRR", "mrr"),
        ("nDCG@5", "ndcg_at_5"),
        ("Source Acc@1", "source_accuracy_at_1"),
        ("Source Acc@5", "source_accuracy_at_5"),
        ("Page Provenance Acc", "page_provenance_accuracy"),
        ("Zero-Hit Count (at 5)", "zero_hit_count"),
        ("Mean Latency (ms)", "mean", "latency_ms"),
        ("P95 Latency (ms)", "p95", "latency_ms"),
    ]

    print("\n" + "=" * 80)
    print("                    RETRIEVAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{headers[0]:<25} | {headers[1]:<14} | {headers[2]:<14} | {headers[3]:<14} | {headers[4]:<13}")
    print("-" * 80)

    for item in rows:
        label = item[0]
        key = item[1]
        parent = item[2] if len(item) > 2 else "retrieval_metrics"

        d_val = dense_m[parent][key]
        b_val = bm25_m[parent][key]
        h_val = hybrid_m[parent][key]

        if "Latency" in label or "Count" in label:
            diff_str = f"{h_val - d_val:+0.1f}"
            print(f"{label:<25} | {d_val:<14} | {b_val:<14} | {h_val:<14} | {diff_str:<13}")
        else:
            diff = (h_val - d_val) * 100.0
            diff_str = f"{diff:+0.2f}%"
            print(f"{label:<25} | {d_val * 100:<13.2f}% | {b_val * 100:<13.2f}% | {h_val * 100:<13.2f}% | {diff_str:<13}")

    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChatLab Grounded RAG retrieval.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT_DIR / "data" / "rag" / "evaluation" / "retrieval_queries.jsonl",
        help="Path to evaluation queries dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "data" / "rag" / "evaluation" / "results.json",
        help="Path to output results JSON.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "bm25", "hybrid", "compare"],
        default="compare",
        help="Evaluation mode: dense (baseline), bm25, hybrid, or compare (all three).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K documents to evaluate (default: 5).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion smoothing parameter (default: 60).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"Loading evaluation dataset from {args.dataset}...")
    queries = load_evaluation_dataset(args.dataset)
    print(f"Loaded {len(queries)} queries.")

    # Load FAISS index and embedding model once
    print(f"Loading FAISS index from {settings.RAG_INDEX_DIR}...")
    t_load = time.perf_counter()
    vector_store, manifest = load_and_validate_index(
        settings.RAG_INDEX_DIR, settings.EMBEDDING_MODEL_PATH
    )
    print(
        f"FAISS index loaded in {time.perf_counter() - t_load:.2f}s ({vector_store.index.ntotal} vectors)"
    )

    # Initialize BM25 index if bm25, hybrid, or compare mode
    bm25_index = None
    if args.mode in ("bm25", "hybrid", "compare"):
        print("Building in-memory BM25 index from documents.jsonl...")
        t_bm25 = time.perf_counter()
        bm25_index = LocalBM25Index()
        bm25_index.build_from_documents(settings.RAG_DOCUMENTS_PATH)
        print(f"BM25 index initialized in {time.perf_counter() - t_bm25:.2f}s")

    results_payload: Dict[str, Any] = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": str(args.dataset).replace("\\", "/"),
            "embedding_model": str(settings.EMBEDDING_MODEL_PATH).replace("\\", "/"),
            "total_queries": len(queries),
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
        }
    }

    if args.mode in ("dense", "compare"):
        print("\n--- Running Dense FAISS Retrieval Benchmark ---")
        dense_m, dense_records = run_benchmark(
            "dense", queries, vector_store, top_k=args.top_k
        )
        results_payload["dense_baseline"] = {
            "metrics": dense_m,
            "queries": [asdict(r) for r in dense_records],
        }
        print(f"Dense Recall@1: {dense_m['retrieval_metrics']['recall_at_1']:.4f}")
        print(f"Dense Recall@5: {dense_m['retrieval_metrics']['recall_at_5']:.4f}")
        print(f"Dense MRR:      {dense_m['retrieval_metrics']['mrr']:.4f}")
        print(f"Dense Zero-hits: {dense_m['retrieval_metrics']['zero_hit_count']}")

    if args.mode in ("bm25", "compare"):
        print("\n--- Running BM25 Sparse Retrieval Benchmark ---")
        bm25_m, bm25_records = run_benchmark(
            "bm25", queries, vector_store, bm25_index=bm25_index, top_k=args.top_k
        )
        results_payload["bm25_sparse"] = {
            "metrics": bm25_m,
            "queries": [asdict(r) for r in bm25_records],
        }
        print(f"BM25 Recall@1:  {bm25_m['retrieval_metrics']['recall_at_1']:.4f}")
        print(f"BM25 Recall@5:  {bm25_m['retrieval_metrics']['recall_at_5']:.4f}")
        print(f"BM25 MRR:       {bm25_m['retrieval_metrics']['mrr']:.4f}")
        print(f"BM25 Zero-hits: {bm25_m['retrieval_metrics']['zero_hit_count']}")

    if args.mode in ("hybrid", "compare"):
        print("\n--- Running Hybrid (Dense + BM25 with RRF) Benchmark ---")
        hybrid_m, hybrid_records = run_benchmark(
            "hybrid",
            queries,
            vector_store,
            bm25_index=bm25_index,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
        )
        results_payload["hybrid_rrf"] = {
            "metrics": hybrid_m,
            "queries": [asdict(r) for r in hybrid_records],
        }
        print(f"Hybrid Recall@1: {hybrid_m['retrieval_metrics']['recall_at_1']:.4f}")
        print(f"Hybrid Recall@5: {hybrid_m['retrieval_metrics']['recall_at_5']:.4f}")
        print(f"Hybrid MRR:      {hybrid_m['retrieval_metrics']['mrr']:.4f}")
        print(f"Hybrid Zero-hits: {hybrid_m['retrieval_metrics']['zero_hit_count']}")

    if args.mode == "compare":
        print_comparison_table(
            results_payload["dense_baseline"]["metrics"],
            results_payload["bm25_sparse"]["metrics"],
            results_payload["hybrid_rrf"]["metrics"],
        )

    # Save complete evaluation payload to data/rag/evaluation/results.json
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved complete evaluation results to {args.output}")


if __name__ == "__main__":
    main()
