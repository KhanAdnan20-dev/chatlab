import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

DATA_PATH = ROOT_DIR / "data" / "intents.json"
MODELS_DIR = ROOT_DIR / "models"
MODEL_OUTPUT_PATH = MODELS_DIR / "intent_classifier.joblib"

from backend.app.services.preprocessing import preprocess


def load_dataset(filepath: Path) -> tuple[list[str], list[str]]:
    """Load and preprocess patterns and labels from intents.json."""
    if not filepath.exists():
        raise FileNotFoundError(f"Intents dataset not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: list[str] = []
    labels: list[str] = []

    for item in data.get("intents", []):
        intent_name = item.get("intent")
        patterns = item.get("patterns", [])
        for pattern in patterns:
            prep = preprocess(pattern)
            processed = prep.processed_text or prep.normalized_text
            texts.append(processed)
            labels.append(intent_name)

    return texts, labels


def train_and_evaluate_model() -> None:
    """Train TF-IDF + Logistic Regression Intent Classifier on 24-class AIML dataset."""
    print("=" * 80)
    print("       ChatLab Layer 1: Supervised AIML Intent Classifier Training       ")
    print("=" * 80)

    print(f"\n[1/5] Loading AIML dataset from: {DATA_PATH}")
    texts, labels = load_dataset(DATA_PATH)
    unique_labels = sorted(list(set(labels)))
    print(f"      Total training samples: {len(texts)}")
    print(f"      Total unique intents:   {len(unique_labels)}")

    # 70% Train, 15% Validation, 15% Test Split
    print("\n[2/5] Creating Stratified Splits (70% Train, 15% Val, 15% Test)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    # Split 15% out of 85% (~17.65%) to obtain exact 15% validation of total
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )

    print(f"      Training set size:   {len(X_train)} samples")
    print(f"      Validation set size: {len(X_val)} samples")
    print(f"      Holdout test size:   {len(X_test)} samples")

    # Construct the Feature Union (Word n-grams 1-3 + Sub-word Char n-grams 2-5)
    feature_union = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    analyzer="word",
                    sublinear_tf=True,
                    min_df=1,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    ngram_range=(2, 5),
                    analyzer="char_wb",
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("features", feature_union),
            (
                "clf",
                LogisticRegression(
                    C=8.0,
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Train on X_train and evaluate on Validation set
    print("\n[3/5] Fitting on Training set & Evaluating on Validation set...")
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"      Validation Set Accuracy: {val_acc * 100:.2f}%")

    # Evaluate on Holdout Test Set
    print("\n[4/5] Evaluating on Independent Holdout Test Set (15%)...")
    test_preds = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"      Holdout Test Accuracy:   {test_acc * 100:.2f}%\n")
    print("Classification Report (Holdout Test Set):")
    print("-" * 80)
    print(classification_report(y_test, test_preds, zero_division=0))

    # Identify most commonly confused intent pairs on validation + test
    cm = confusion_matrix(y_test, test_preds, labels=unique_labels)
    confused_pairs = []
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((unique_labels[i], unique_labels[j], int(cm[i, j])))

    if confused_pairs:
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        print("\nTop Confused Intent Pairs (Actual -> Predicted):")
        for actual, pred, count in confused_pairs[:5]:
            print(f"  • '{actual}' -> '{pred}' ({count} instances)")

    # Fit final pipeline on 100% of the dataset and save artifact
    print(f"\n[5/5] Fitting final pipeline on entire dataset ({len(texts)} samples)...")
    pipeline.fit(texts, labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"      [OK] Serialized model saved to: {MODEL_OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    train_and_evaluate_model()
