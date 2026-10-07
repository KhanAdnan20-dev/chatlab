# ChatLab Dataset Architecture & Specifications

This document outlines the design, schema, and methodology behind the **dual-layer dataset architecture** developed for ChatLab.

---

## 1. Architectural Philosophy: Dual-Layer Dataset Separation

ChatLab separates conversational data into two complementary layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CHATLAB DATA ARCHITECTURE                     │
├────────────────────────────────────┬────────────────────────────────────┤
│ LAYER 1 DATASET (intents.json)     │ LAYER 2 DATASET (qa_dataset.jsonl) │
├────────────────────────────────────┼────────────────────────────────────┤
│ • Objective: What does user want?  │ • Objective: How to answer deeply? │
│ • Supervised Intent Classification │ • Supervised QLoRA Fine-Tuning     │
│ • Deterministic / Fast Mapping     │ • Generative Deep Explanations     │
│ • 24 AIML Intents, 768 Patterns    │ • 3,000 Instruction-Response Pairs │
│ • Sub-millisecond NLP Inference    │ • Autoregressive LLM Reasoning     │
└────────────────────────────────────┴────────────────────────────────────┘
```

---

## 2. Intent vs. Entity (The NLU Foundation)

- **Intent**: The functional goal or conversational purpose of the user's utterance (e.g., `concept_definition`, `formula_request`, `concept_comparison`, `start_quiz`).
- **Entity**: The canonical domain object, algorithm, or parameter referenced in the utterance (e.g., `decision_tree`, `backpropagation`, `f1_score`, `learning_rate`).

**Example**:
> *"Can you give me the formula for precision and recall in classification?"*
- **Intent**: `formula_request`
- **Entities**: `precision` (canonical: `precision`), `recall` (canonical: `recall`), `classification` (canonical: `classification`).

---

## 3. Layer 1 Intents Dataset (`data/intents.json`)

### 3.1 Structure
- **Total Intents**: 24 Canonical Categories
- **Total Training Utterances**: 768 annotated patterns (32 per intent)
- **Response Templates**: 72 curated response templates

### 3.2 The 24 Intents
1. **Conversational (7)**: `greeting`, `goodbye`, `thanks`, `bot_identity`, `capabilities`, `help`, `unknown`.
2. **AIML Question Types (11)**: `concept_definition`, `concept_explanation`, `concept_comparison`, `example_request`, `formula_request`, `algorithm_steps`, `advantages_disadvantages`, `applications`, `limitations`, `code_explanation`, `problem_solving`.
3. **Interactive Quiz (6)**: `start_quiz`, `quiz_topic`, `quiz_difficulty`, `quiz_answer`, `quiz_score`, `quiz_explanation`.

---

## 4. Layer 2 Qwen Fine-Tuning Dataset (`data/qa_dataset.jsonl`)

### 4.1 Schema
Each line in `data/qa_dataset.jsonl` follows standard instruction format:
```json
{
  "instruction": "What is the difference between supervised and unsupervised learning?",
  "context": "",
  "response": "Supervised learning utilizes labeled pairs (X, y) to minimize loss..."
}
```

### 4.2 Category Distribution (3,000 Total Pairs)
1. **Concept Definitions**: ~500 examples
2. **Concept Explanations**: ~500 examples
3. **Concept Comparisons**: ~350 examples
4. **Examples / Intuition**: ~250 examples
5. **Formulas / Metrics**: ~200 examples
6. **Algorithm / Procedure Steps**: ~250 examples
7. **Advantages / Disadvantages**: ~200 examples
8. **Applications**: ~150 examples
9. **Problem Solving (Math & Metrics)**: ~250 examples
10. **Code Explanations (Python, Sklearn, PyTorch)**: ~150 examples
11. **Quiz Generation**: ~100 examples
12. **Quiz Interaction & Answer Reasoning**: ~100 examples

### 4.3 Difficulty Breakdown
- **Beginner (35%)**: Introductory definitions, basic intuition, core taxonomy.
- **Intermediate (45%)**: Mathematical loss formulations, algorithm comparisons, trade-off analysis.
- **Advanced (20%)**: Multi-step mathematical derivations, vanishing gradients, transformer self-attention projections.

### 4.4 Dataset Splits
- **Train Split (`qa_train.jsonl`)**: 2,400 samples (80.0%)
- **Validation Split (`qa_validation.jsonl`)**: 300 samples (10.0%)
- **Test Split (`qa_test.jsonl`)**: 300 samples (10.0%)
- **Leakage**: 0% overlap between train and validation/test subsets.

---

## 5. Dataset Validation & Quality Assurance

Validated automatically via `python scripts/validate_dataset.py` and `python scripts/validate_qa_dataset.py`:
- [x] 100% Valid JSON and JSONL formatting.
- [x] Zero duplicate instructions.
- [x] Zero empty instruction/response fields.
- [x] Zero data leakage across splits.
- [x] All 70 canonical AIML concepts and 259 aliases validated.
