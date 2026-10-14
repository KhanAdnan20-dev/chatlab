#!/usr/bin/env python3
"""
ChatLab Grounded RAG Pipeline — Ingestion Script.

Recursively discovers raw PDFs, extracts text page-by-page with PyMuPDFLoader,
normalizes whitespace, applies deterministic LangChain RecursiveCharacterTextSplitter,
enriches metadata using manifests/sources.json, exports documents.jsonl, and builds
a normalized FAISS inner-product vector store.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
    ChunkRecord,
    PDFExtractionError,
    build_faiss_index,
    chunk_document_pages,
    export_documents_jsonl,
    extract_pdf_pages,
    get_optimal_device,
    load_sources_manifest,
    match_source_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_rag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into ChatLab RAG documents.jsonl and FAISS vector index."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT_DIR / "data" / "rag" / "raw",
        help="Directory containing raw PDF documents (default: data/rag/raw)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT_DIR / "data" / "rag" / "manifests" / "sources.json",
        help="Path to sources.json manifest (default: data/rag/manifests/sources.json)",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=ROOT_DIR / "data" / "rag" / "processed" / "documents.jsonl",
        help="Output path for processed documents JSONL (default: data/rag/processed/documents.jsonl)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT_DIR / "data" / "rag" / "indexes" / "faiss",
        help="Output directory for FAISS vector index (default: data/rag/indexes/faiss)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT_DIR / "models" / "all-MiniLM-L6-v2",
        help="Path to local embedding model (default: models/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=220,
        help="Target chunk size in words (default: 220)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help="Chunk overlap in words (default: 40)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding generation batch size (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Embedding device: 'cuda' or 'cpu' (default: autodetect CUDA)",
    )
    parser.add_argument(
        "--limit-pdfs",
        type=int,
        default=None,
        help="Optional limit on number of PDFs to process (for testing)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional limit on pages per PDF to process (for testing)",
    )
    return parser.parse_args()


def run_ingestion(args: argparse.Namespace) -> int:
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("Starting ChatLab RAG Ingestion Pipeline")
    logger.info("=================================================================")
    logger.info("Raw PDF Directory:     %s", args.raw_dir)
    logger.info("Sources Manifest:      %s", args.manifest_path)
    logger.info("Output JSONL:          %s", args.output_jsonl)
    logger.info("Index Directory:       %s", args.index_dir)
    logger.info("Embedding Model:       %s", args.model_path)
    logger.info("Chunk Size:            ~%d words", args.chunk_size)
    logger.info("Chunk Overlap:         ~%d words", args.chunk_overlap)
    logger.info("Batch Size:            %d", args.batch_size)

    device = get_optimal_device(args.device)
    logger.info("Hardware Device:       %s", device)

    # 1. Discover raw PDFs recursively
    if not args.raw_dir.exists():
        logger.error("Raw PDF directory does not exist: %s", args.raw_dir)
        return 1

    discovered_pdfs = sorted(args.raw_dir.rglob("*.pdf"))
    if not discovered_pdfs:
        logger.warning("No PDF files discovered in %s", args.raw_dir)
        return 0

    logger.info("Discovered %d PDF file(s) in %s", len(discovered_pdfs), args.raw_dir)
    if args.limit_pdfs:
        discovered_pdfs = discovered_pdfs[: args.limit_pdfs]
        logger.info("Limiting processing to first %d PDF(s)", len(discovered_pdfs))

    # 2. Load sources manifest
    sources_manifest = load_sources_manifest(args.manifest_path)
    logger.info("Loaded %d verified source entries from manifest", len(sources_manifest))

    # 3. Process each PDF
    all_chunks: List[ChunkRecord] = []
    processed_count = 0
    failed_pdfs: List[Dict[str, str]] = []
    total_pages_extracted = 0
    total_pages_skipped = 0
    skipped_ocr_pages_detail: List[Dict[str, Any]] = []

    for idx, pdf_path in enumerate(discovered_pdfs, start=1):
        rel_path = pdf_path.relative_to(ROOT_DIR).as_posix()
        logger.info("[%d/%d] Processing '%s'...", idx, len(discovered_pdfs), pdf_path.name)

        try:
            valid_pages, skipped_pages, pdf_meta = extract_pdf_pages(
                pdf_path=pdf_path,
                max_pages=args.max_pages,
            )
        except PDFExtractionError as exc:
            logger.error("Failed to extract '%s': %s (continuing with next file)", pdf_path.name, exc)
            failed_pdfs.append({"file": pdf_path.name, "path": rel_path, "reason": str(exc)})
            continue
        except Exception as exc:
            logger.error("Unexpected error on '%s': %s (continuing with next file)", pdf_path.name, exc)
            failed_pdfs.append({"file": pdf_path.name, "path": rel_path, "reason": f"Unexpected: {exc}"})
            continue

        pages_extracted = len(valid_pages)
        pages_skipped = len(skipped_pages)
        total_pages_extracted += pages_extracted
        total_pages_skipped += pages_skipped

        if pages_skipped > 0:
            logger.warning(
                "  '%s': %d page(s) had no extractable text (OCR required if scanned). Skipped pages: %s",
                pdf_path.name,
                pages_skipped,
                skipped_pages[:10] if len(skipped_pages) <= 10 else f"{skipped_pages[:10]}... (+{pages_skipped - 10} more)",
            )
            skipped_ocr_pages_detail.append({
                "file": pdf_path.name,
                "skipped_count": pages_skipped,
                "pages": skipped_pages,
            })

        if not valid_pages:
            logger.warning("  No valid text extracted from '%s'. Skipping chunk generation.", pdf_path.name)
            continue

        # Match and enrich metadata
        source_meta = match_source_metadata(
            pdf_path=pdf_path,
            project_root=ROOT_DIR,
            sources_manifest=sources_manifest,
            pdf_title=pdf_meta.get("title"),
        )

        logger.info(
            "  Source matched: id='%s', title='%s', subject='%s', url=%s",
            source_meta["source_id"],
            source_meta["source_title"],
            source_meta["subject"],
            source_meta["source_url"] or "None (unverified)",
        )

        # Chunk document
        chunks = chunk_document_pages(
            valid_pages=valid_pages,
            source_metadata=source_meta,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        logger.info("  Extracted %d pages -> Generated %d chunks", pages_extracted, len(chunks))
        all_chunks.extend(chunks)
        processed_count += 1

    if not all_chunks:
        logger.error("No chunks were generated across all discovered PDFs.")
        return 1

    # 4. Write documents.jsonl
    logger.info("Exporting %d total chunks to %s...", len(all_chunks), args.output_jsonl)
    export_documents_jsonl(all_chunks, args.output_jsonl)
    logger.info("Successfully wrote and validated %s", args.output_jsonl)

    # 5. Build FAISS index
    sources_summary = {
        "total_pdfs_discovered": len(discovered_pdfs),
        "total_pdfs_processed": processed_count,
        "failed_pdfs_count": len(failed_pdfs),
        "total_pages_extracted": total_pages_extracted,
        "total_pages_skipped_ocr_required": total_pages_skipped,
        "total_chunks_generated": len(all_chunks),
    }

    logger.info("Building FAISS index at %s...", args.index_dir)
    build_faiss_index(
        chunks=all_chunks,
        model_path=args.model_path,
        index_dir=args.index_dir,
        batch_size=args.batch_size,
        device=device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sources_summary=sources_summary,
    )

    elapsed_time = time.time() - start_time

    # 6. End-of-run Report
    print("\n" + "=" * 70)
    print("           CHATLAB RAG INGESTION PIPELINE REPORT")
    print("=" * 70)
    print(f"Discovered PDFs:                 {len(discovered_pdfs)}")
    print(f"Successfully Processed PDFs:     {processed_count}")
    print(f"Failed PDFs:                     {len(failed_pdfs)}")
    print(f"Total Pages Extracted:           {total_pages_extracted}")
    print(f"Pages with No Text (OCR Needed): {total_pages_skipped}")
    print(f"Generated Chunks:                {len(all_chunks)}")
    print(f"Indexed Vectors:                 {len(all_chunks)}")
    print(f"Embedding Model:                 {args.model_path.name}")
    print(f"Hardware Device:                 {device.upper()}")
    print(f"Total Execution Time:            {elapsed_time:.2f} seconds")
    print("-" * 70)
    print(f"Output Documents JSONL:          {args.output_jsonl}")
    print(f"Output FAISS Index Directory:    {args.index_dir}")
    print("=" * 70)

    if failed_pdfs:
        print("\nFailed PDFs Summary:")
        for failed in failed_pdfs:
            print(f"  - {failed['file']}: {failed['reason']}")

    if skipped_ocr_pages_detail:
        print("\nPages with No Extractable Text (OCR Required if Scanned):")
        for item in skipped_ocr_pages_detail:
            pg_str = ", ".join(map(str, item["pages"][:10]))
            if item["skipped_count"] > 10:
                pg_str += f"... (+{item['skipped_count'] - 10} more)"
            print(f"  - {item['file']}: {item['skipped_count']} page(s) (pages: {pg_str})")

    print("\nIngestion completed successfully!\n")
    return 0


def main() -> None:
    args = parse_args()
    exit_code = run_ingestion(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
