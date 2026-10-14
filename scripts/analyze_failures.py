#!/usr/bin/env python3
"""
Summarize failure cases from results.json.
"""

import json
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

with open("data/rag/evaluation/results.json", "r", encoding="utf-8") as f:
    res = json.load(f)

dense = res["dense_baseline"]["metrics"]
print("=== DENSE ERROR DISTRIBUTION ===")
for k, v in dense.get("error_distribution", {}).items():
    print(f"  {k}: {v}")

print(f"\nTotal missed queries (at 5): {len(dense['missed_queries'])}")
print("\n=== SAMPLE FAILURE CASES ===")
for m in dense["missed_queries"]:
    qid = m["query_id"]
    cat = m["category"]
    q = m["query"]
    err = m["error_class"]
    top = m["top_retrieved"][0] if m["top_retrieved"] else None
    print(f"\n[{qid}] ({cat}) '{q}'")
    print(f"  Error class: {err}")
    print(f"  Expected: {m['expected_sources']}")
    if top:
        print(f"  Retrieved top-1: {top['chunk_id']} ({top['file_name']} pp. {top['pages']}, score: {top['score']})")

print("\n=== UNSUPPORTED NEGATIVE QUERIES GATING ===")
dense_queries = res["dense_baseline"]["queries"]
unsupported = [q for q in dense_queries if not q["is_supported"]]
for u in unsupported:
    qid = u["query_id"]
    q = u["query"]
    gated = u["is_gated_properly"]
    max_score = max(u["retrieved_scores"]) if u["retrieved_scores"] else 0.0
    print(f"[{qid}] Gated properly: {gated} (max_score: {max_score:.4f}) | '{q}'")
