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
DATA_DIR = ROOT_DIR / "data"
QA_DATASET_PATH = DATA_DIR / "qa_dataset.jsonl"
QA_TRAIN_PATH = DATA_DIR / "qa_train.jsonl"
QA_VAL_PATH = DATA_DIR / "qa_validation.jsonl"
QA_TEST_PATH = DATA_DIR / "qa_test.jsonl"


def validate_qa_files() -> bool:
    print("=" * 80)
    print("             ChatLab Phase 7: Qwen Fine-Tuning QA Dataset Validation             ")
    print("=" * 80)

    all_passed = True

    # 1. Validate main dataset file
    print(f"\n[1/4] Validating Full Dataset: {QA_DATASET_PATH.name}")
    if not QA_DATASET_PATH.exists():
        print("  [FAIL] ERROR: qa_dataset.jsonl does not exist!")
        return False

    records = []
    seen_instructions = set()
    duplicate_instructions = 0
    empty_fields = 0
    short_responses = 0

    with open(QA_DATASET_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
            except Exception as e:
                print(f"  [FAIL] Malformed JSON on line {idx}: {e}")
                all_passed = False
                continue

            inst = data.get("instruction", "").strip()
            resp = data.get("response", "").strip()
            data.get("context", "")

            if not inst or not resp:
                empty_fields += 1
                all_passed = False

            if inst.lower() in seen_instructions:
                duplicate_instructions += 1
                all_passed = False
            seen_instructions.add(inst.lower())

            if len(resp) < 20:
                short_responses += 1

            records.append(data)

    print(f"  [OK] Total valid JSONL records: {len(records)}")
    if duplicate_instructions == 0:
        print("  [OK] Zero duplicate instructions detected.")
    else:
        print(f"  [FAIL] {duplicate_instructions} duplicate instructions found!")

    if empty_fields == 0:
        print("  [OK] Zero empty instruction/response fields detected.")
    else:
        print(f"  [FAIL] {empty_fields} records have empty fields!")

    # 2. Validate train/val/test files and leakage
    print(f"\n[2/4] Validating Split Files & Checking Data Leakage...")
    
    def load_split(p: Path) -> list[dict]:
        items = []
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        return items

    train_recs = load_split(QA_TRAIN_PATH)
    val_recs = load_split(QA_VAL_PATH)
    test_recs = load_split(QA_TEST_PATH)

    print(f"  • Training set records:   {len(train_recs)}")
    print(f"  • Validation set records: {len(val_recs)}")
    print(f"  • Test set records:       {len(test_recs)}")

    if len(train_recs) != 2400 or len(val_recs) != 300 or len(test_recs) != 300:
        print(f"  [WARN] Expected 2400/300/300 counts.")

    train_insts = {r["instruction"].lower() for r in train_recs}
    val_insts = {r["instruction"].lower() for r in val_recs}
    test_insts = {r["instruction"].lower() for r in test_recs}

    val_leakage = train_insts.intersection(val_insts)
    test_leakage = train_insts.intersection(test_insts)
    val_test_leakage = val_insts.intersection(test_insts)

    if not val_leakage and not test_leakage and not val_test_leakage:
        print("  [OK] Zero data leakage across train, validation, and test splits.")
    else:
        print(f"  [FAIL] Data leakage detected: val_leakage={len(val_leakage)}, test_leakage={len(test_leakage)}")
        all_passed = False

    # 3. Response Length Analysis
    print(f"\n[3/4] Response Character & Word Length Statistics:")
    word_counts = [len(r["response"].split()) for r in records]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    print(f"  • Average response length: {avg_words:.1f} words")
    print(f"  • Min response length:     {min_words} words")
    print(f"  • Max response length:     {max_words} words")

    print(f"\n[4/4] Dataset Validation Status:")
    print("=" * 80)
    if all_passed:
        print("             [PASSED] QA DATASET FULLY VALIDATED AND READY FOR QLORA             ")
    else:
        print("             [FAILED] SOME QA VALIDATION CHECKS FAILED (SEE ABOVE)              ")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = validate_qa_files()
    sys.exit(0 if success else 1)
