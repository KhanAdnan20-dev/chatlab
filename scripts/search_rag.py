#!/usr/bin/env python3
"""
ChatLab Grounded RAG Pipeline — Search & Retrieval Test Script.

Validates the local FAISS index, manifest, embedding model, and metadata,
then performs semantic inner-product retrieval for a natural-language query.
"""

import argparse
import sys
from pathlib import Path
from typing import List

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

from backend.app.rag.pipeline import (
    IndexValidationError,
    SearchResult,
    load_and_validate_index,
    search_index,
    validate_documents_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search ChatLab RAG FAISS vector store with a natural-language query."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Natural language query string (e.g., 'How does multi-head attention work?')",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        dest="query_opt",
        default=None,
        help="Alternative flag to specify the query string",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT_DIR / "data" / "rag" / "indexes" / "faiss",
        help="Path to FAISS index directory (default: data/rag/indexes/faiss)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT_DIR / "models" / "all-MiniLM-L6-v2",
        help="Path to local embedding model (default: models/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--documents-path",
        type=Path,
        default=ROOT_DIR / "data" / "rag" / "processed" / "documents.jsonl",
        help="Path to processed documents.jsonl (default: data/rag/processed/documents.jsonl)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Embedding device: 'cuda' or 'cpu' (default: autodetect)",
    )
    parser.add_argument(
        "--snippet-len",
        type=int,
        default=320,
        help="Maximum character length of the text snippet to display (default: 320)",
    )
    return parser.parse_args()


def display_results(query: str, results: List[SearchResult], manifest: dict) -> None:
    print("\n" + "=" * 78)
    print(f"Query: \"{query}\"")
    print(f"Index: {manifest.get('embedding_model')} | Total Chunks: {manifest.get('total_chunks')} | Returned: {len(results)}")
    print("=" * 78 + "\n")

    if not results:
        print("No matching literature chunks found above threshold.")
        return

    for res in results:
        page_str = f"p. {res.page_start}" if res.page_start == res.page_end else f"pp. {res.page_start}-{res.page_end}"
        print(f"Rank {res.rank}  |  Similarity Score: {res.score:.4f}  |  Subject: {res.subject}")
        print(f"Source: {res.source_title}")
        print(f"File:   {res.file_name} ({page_str})")
        if res.source_url:
            print(f"URL:    {res.source_url}")
        print(f"Chunk:  {res.chunk_id}")
        print("-" * 78)
        print("Snippet:")
        for line in res.snippet.split("\n"):
            print(f"  {line}")
        print("=" * 78 + "\n")


def main() -> None:
    args = parse_args()
    query = args.query or args.query_opt

    if not query or not query.strip():
        print("Error: Please provide a search query either as a positional argument or via --query/-q.", file=sys.stderr)
        print("Example: python scripts/search_rag.py \"How does backpropagation calculate gradients?\"", file=sys.stderr)
        sys.exit(1)

    query = query.strip()

    # Pre-flight validation of all artifacts
    print("[1/3] Validating RAG artifacts...")
    if not args.documents_path.exists():
        print(f"Error: Processed documents file not found at {args.documents_path}.", file=sys.stderr)
        print("Please run 'python scripts/ingest_rag.py' first.", file=sys.stderr)
        sys.exit(1)

    try:
        doc_count = validate_documents_jsonl(args.documents_path)
        print(f"  - Verified {args.documents_path.name} ({doc_count} valid records)")
    except Exception as exc:
        print(f"Error: Invalid documents.jsonl: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.index_dir.exists():
        print(f"Error: FAISS index directory not found at {args.index_dir}.", file=sys.stderr)
        print("Please run 'python scripts/ingest_rag.py' first.", file=sys.stderr)
        sys.exit(1)

    print("[2/3] Loading trusted FAISS index and local embedding model...")
    try:
        vector_store, manifest = load_and_validate_index(
            index_dir=args.index_dir,
            model_path=args.model_path,
            device=args.device,
        )
        print(f"  - Loaded FAISS index ({vector_store.index.ntotal} vectors, dimension {vector_store.index.d})")
    except IndexValidationError as exc:
        print(f"Index Validation Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to load FAISS index: {exc}", file=sys.stderr)
        sys.exit(1)

    # Perform retrieval
    print(f"[3/3] Retrieving top-{args.top_k} results for: \"{query}\"...")
    try:
        results = search_index(
            vector_store=vector_store,
            query=query,
            top_k=args.top_k,
            snippet_len=args.snippet_len,
        )
    except Exception as exc:
        print(f"Retrieval error: {exc}", file=sys.stderr)
        sys.exit(1)

    display_results(query, results, manifest)


if __name__ == "__main__":
    main()
