import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INTENTS_PATH = ROOT_DIR / "data" / "intents.json"
TOPICS_PATH = ROOT_DIR / "data" / "aiml_topics.json"
QA_DATASET_PATH = ROOT_DIR / "data" / "qa_dataset.jsonl"


def print_statistics():
    print("=" * 70)
    print("                 ChatLab Dataset Statistics Summary                 ")
    print("=" * 70)

    if INTENTS_PATH.exists():
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        intents = data.get("intents", [])
        total_p = sum(len(i.get("patterns", [])) for i in intents)
        total_r = sum(len(i.get("responses", [])) for i in intents)
        print(f"\n[Layer 1 Intents Dataset]")
        print(f"  • Total Intents:            {len(intents)}")
        print(f"  • Total Training Patterns:  {total_p}")
        print(f"  • Average Patterns/Intent:  {total_p / len(intents):.1f}")
        print(f"  • Total Response Templates: {total_r}")

    if TOPICS_PATH.exists():
        with open(TOPICS_PATH, "r", encoding="utf-8") as f:
            tdata = json.load(f)
        topics = tdata.get("topics", {})
        total_aliases = sum(len(a) for a in topics.values())
        print(f"\n[AIML Entity & Topic Dictionary]")
        print(f"  • Canonical Topics:         {len(topics)}")
        print(f"  • Total Topic Aliases:      {total_aliases}")

    if QA_DATASET_PATH.exists():
        count = 0
        with open(QA_DATASET_PATH, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        print(f"\n[Layer 2 Qwen QA Fine-Tuning Dataset]")
        print(f"  • Total Instruction Pairs:  {count}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_statistics()
