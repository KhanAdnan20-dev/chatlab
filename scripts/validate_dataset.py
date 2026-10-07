import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
INTENTS_PATH = ROOT_DIR / "data" / "intents.json"
TOPICS_PATH = ROOT_DIR / "data" / "aiml_topics.json"

EXPECTED_24_INTENTS = [
    "greeting",
    "goodbye",
    "thanks",
    "bot_identity",
    "capabilities",
    "help",
    "unknown",
    "concept_definition",
    "concept_explanation",
    "concept_comparison",
    "example_request",
    "formula_request",
    "algorithm_steps",
    "advantages_disadvantages",
    "applications",
    "limitations",
    "code_explanation",
    "problem_solving",
    "start_quiz",
    "quiz_topic",
    "quiz_difficulty",
    "quiz_answer",
    "quiz_score",
    "quiz_explanation",
]


def validate_datasets() -> bool:
    print("=" * 75)
    print("             ChatLab Phase 3: Dataset Integrity & Validation             ")
    print("=" * 75)

    all_passed = True

    # 1. Validate intents.json
    print(f"\n[1/2] Checking Intents Dataset: {INTENTS_PATH.name}")
    if not INTENTS_PATH.exists():
        print("  [FAIL] ERROR: intents.json file does not exist!")
        return False

    try:
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            intents_data = json.load(f)
        print("  [OK] Valid JSON syntax.")
    except Exception as e:
        print(f"  [FAIL] Invalid JSON in intents.json: {e}")
        return False

    intents_list = intents_data.get("intents", [])
    found_intent_names = [item.get("intent") for item in intents_list]

    # Check 24 intents
    missing_intents = set(EXPECTED_24_INTENTS) - set(found_intent_names)
    if missing_intents:
        print(f"  [FAIL] Missing required intents: {missing_intents}")
        all_passed = False
    else:
        print(f"  [OK] All 24 canonical AIML intents present.")

    # Check duplicates and pattern counts
    total_patterns = 0
    seen_global_patterns: dict[str, str] = {}
    cross_intent_duplicates: list[tuple[str, str, str]] = []

    for item in intents_list:
        name = item.get("intent", "UNKNOWN")
        patterns = item.get("patterns", [])
        responses = item.get("responses", [])

        if len(patterns) < 30:
            print(f"  [WARN] Intent '{name}' has {len(patterns)} patterns (< 30 target).")
            all_passed = False

        if not responses or len(responses) == 0:
            print(f"  [FAIL] Error: Intent '{name}' has no response templates.")
            all_passed = False

        seen_local = set()
        for p in patterns:
            norm_p = p.strip().lower()
            if not norm_p:
                print(f"  [FAIL] Empty pattern found in intent '{name}'!")
                all_passed = False

            if norm_p in seen_local:
                print(f"  [FAIL] Internal duplicate within intent '{name}': \"{p}\"")
                all_passed = False
            seen_local.add(norm_p)

            if norm_p in seen_global_patterns:
                cross_intent_duplicates.append((norm_p, seen_global_patterns[norm_p], name))
            else:
                seen_global_patterns[norm_p] = name

            total_patterns += 1

    if cross_intent_duplicates:
        print(f"  [FAIL] Cross-intent duplicates found ({len(cross_intent_duplicates)}):")
        for p, i1, i2 in cross_intent_duplicates:
            print(f"     - \"{p}\" appears in '{i1}' and '{i2}'")
        all_passed = False
    else:
        print(f"  [OK] Zero cross-intent duplicate utterances found across {total_patterns} total patterns.")

    # 2. Validate aiml_topics.json
    print(f"\n[2/2] Checking AIML Topics & Entity Database: {TOPICS_PATH.name}")
    if not TOPICS_PATH.exists():
        print("  [FAIL] ERROR: aiml_topics.json file does not exist!")
        return False

    try:
        with open(TOPICS_PATH, "r", encoding="utf-8") as f:
            topics_data = json.load(f)
        print("  [OK] Valid JSON syntax.")
    except Exception as e:
        print(f"  [FAIL] Invalid JSON in aiml_topics.json: {e}")
        return False

    topics = topics_data.get("topics", {})
    print(f"  [OK] Total canonical topics defined: {len(topics)}")

    total_aliases = 0
    alias_seen = {}
    for canonical, aliases in topics.items():
        if not isinstance(aliases, list) or len(aliases) == 0:
            print(f"  [FAIL] Canonical topic '{canonical}' has no aliases list!")
            all_passed = False
        for a in aliases:
            total_aliases += 1
            low_a = a.strip().lower()
            if low_a in alias_seen:
                print(f"  [WARN] Duplicate alias '{low_a}' in '{alias_seen[low_a]}' and '{canonical}'")
            alias_seen[low_a] = canonical

    print(f"  [OK] Total topic aliases mapped: {total_aliases}")

    print("\n" + "=" * 75)
    if all_passed:
        print("                  [PASSED] ALL DATASET VALIDATION CHECKS PASSED                  ")
    else:
        print("               [FAILED] SOME VALIDATION CHECKS FAILED (SEE ABOVE)               ")
    print("=" * 75)

    return all_passed


if __name__ == "__main__":
    success = validate_datasets()
    sys.exit(0 if success else 1)
