import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.chatbot import chatbot_service

TEST_QUERIES = [
    "hello",
    "thanks",
    "what can you do?",
    "help me study for my exam",
    "teach me recursion",
    "explain backpropagation mathematically",
    "compare CNN and transformers",
    "what should I do about my semester workload?",
    "I have a Python exam on Monday and need study tips",
    "asdfghjk 998877 random out of domain gibberish string",
]


def run_fuzzy_router_experiments() -> None:
    print("=" * 95)
    print("        ChatLab Phase 2: Mamdani Fuzzy Logic Decision Router Experiments        ")
    print("=" * 95)

    for idx, query in enumerate(TEST_QUERIES, 1):
        result = chatbot_service.process(query)

        print(f"\n[Case {idx}] Query: \"{query}\"")
        print("-" * 95)
        print(f"  • Classified Intent:     {result.intent}")
        print(f"  • Intent Confidence (C):  {result.confidence:.4f}")
        print(f"  • Semantic Match (S):     {result.similarity_score:.4f} (Pattern: \"{result.matched_pattern}\")")
        print(f"  • Query Complexity (L):   {result.complexity_score:.4f}")
        print(f"  • Entity Density (E):     {result.entity_density:.4f} ({len(result.entities)} entities: {[e.text for e in result.entities]})")
        print(f"  • Fuzzy Activation (A):   {result.activation_score:.4f}")
        print(f"  • Selected Layer:         {result.layer.upper()}")
        print(f"  • Routing Reason:         {result.routing_reason}")
        print(f"  • Bot Response:           \"{result.response}\"")

    print("\n" + "=" * 95)
    print("                      All Router Experiment Cases Evaluated                      ")
    print("=" * 95)


if __name__ == "__main__":
    run_fuzzy_router_experiments()
