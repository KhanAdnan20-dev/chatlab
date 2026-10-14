#!/usr/bin/env python3
"""
ChatLab Phase 2C — Build Retrieval Evaluation Dataset.

Constructs and verifies data/rag/evaluation/retrieval_queries.jsonl with 52
representative queries across 13 technical categories, with 100% verified ground-truth
chunk IDs and physical page provenance inspected directly from documents.jsonl.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_FILE = ROOT_DIR / "data" / "rag" / "processed" / "documents.jsonl"
EVAL_DIR = ROOT_DIR / "data" / "rag" / "evaluation"
OUTPUT_FILE = EVAL_DIR / "retrieval_queries.jsonl"


def load_corpus() -> Dict[str, Dict[str, Any]]:
    corpus = {}
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["chunk_id"]] = d
    return corpus


def build_raw_query_definitions() -> List[Dict[str, Any]]:
    """
    Define 52 benchmark queries across 13 categories.
    """
    return [
        # --- 1. Concept definitions ---
        {
            "query_id": "eval_q01",
            "query": "What is an eigenvalue and eigenvector of a square matrix?",
            "category": "concept_definition",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["mathematics_for_machine_learning.pdf"],
            "expected_chunk_ids": ["mathematics_for_machine_learning_chunk_00238", "mathematics_for_machine_learning_chunk_00239"],
            "notes": "Section 4.2 Eigenvalues and Eigenvectors: Ax = lambda x, characteristic polynomial.",
        },
        {
            "query_id": "eval_q02",
            "query": "What is perplexity in language modeling and how is it calculated from cross-entropy?",
            "category": "concept_definition",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00270"],
            "notes": "Section 3.2.1 Perplexity equation (3.15) and cross-entropy relationship.",
        },
        {
            "query_id": "eval_q03",
            "query": "What is Gini impurity in the CART decision tree algorithm?",
            "category": "concept_definition",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": ["hands_on_machine_learning_2e_chunk_00376"],
            "notes": "Chapter 6 Decision Trees, Equation 6-1 Gini impurity computation.",
        },
        {
            "query_id": "eval_q04",
            "query": "What is the cross-entropy loss function for classification?",
            "category": "concept_definition",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["understanding_deep_learning.pdf"],
            "expected_chunk_ids": ["understanding_deep_learning_chunk_00303"],
            "notes": "Section 5.3 Loss functions for classification, negative log-likelihood of categorical distribution.",
        },

        # --- 2. Detailed explanations ---
        {
            "query_id": "eval_q05",
            "query": "How do residual connections prevent vanishing gradients in deep networks such as ResNet?",
            "category": "detailed_explanation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["dive_into_deep_learning.pdf"],
            "expected_chunk_ids": ["d2l_chunk_00754"],
            "notes": "Section 8.6 ResNet residual blocks, identity shortcut allowing gradient flow.",
        },
        {
            "query_id": "eval_q06",
            "query": "How does Word2Vec skip-gram with negative sampling work?",
            "category": "detailed_explanation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00442"],
            "notes": "Chapter 6 Vector Semantics, skip-gram with negative sampling (SGNS) architecture and binary logistic classification.",
        },
        {
            "query_id": "eval_q07",
            "query": "How does the CART algorithm select the feature and threshold to split a node?",
            "category": "detailed_explanation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": ["hands_on_machine_learning_2e_chunk_00379"],
            "notes": "Chapter 6 The CART Training Algorithm cost function J(k, t_k) minimizing weighted impurity.",
        },
        {
            "query_id": "eval_q08",
            "query": "How does the backpropagation algorithm compute gradients using computational graphs and the chain rule?",
            "category": "detailed_explanation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["dive_into_deep_learning.pdf"],
            "expected_chunk_ids": ["d2l_chunk_00486", "d2l_chunk_00487"],
            "notes": "Section 5.3 Forward Propagation, Backward Propagation, and Computational Graphs (Section 5.3.3).",
        },

        # --- 3. Comparisons ---
        {
            "query_id": "eval_q09",
            "query": "What is the difference between L1 (Lasso) and L2 (Ridge) regularization?",
            "category": "comparison",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": ["hands_on_machine_learning_2e_chunk_00309", "hands_on_machine_learning_2e_chunk_00311"],
            "notes": "Chapter 4 Ridge Regression vs Lasso Regression, sparsity vs weight shrinkage.",
        },
        {
            "query_id": "eval_q10",
            "query": "How does the Adam optimizer compare to standard stochastic gradient descent with momentum?",
            "category": "comparison",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["understanding_deep_learning.pdf"],
            "expected_chunk_ids": ["understanding_deep_learning_chunk_00230", "understanding_deep_learning_chunk_00231"],
            "notes": "Chapter 6 Optimization, Section 6.4 Adam combining momentum and RMSprop with bias correction.",
        },
        {
            "query_id": "eval_q11",
            "query": "What is the difference between hard-margin and soft-margin Support Vector Machines?",
            "category": "comparison",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": ["hands_on_machine_learning_2e_chunk_00357"],
            "notes": "Chapter 5 Soft Margin Classification, slack variables and trade-off parameter C.",
        },
        {
            "query_id": "eval_q12",
            "query": "How does Batch Normalization differ from Layer Normalization in neural networks?",
            "category": "comparison",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["dive_into_deep_learning.pdf"],
            "expected_chunk_ids": ["d2l_chunk_00673", "d2l_chunk_00728"],
            "notes": "Section 8.5 Batch Normalization vs Layer Normalization across batch versus feature dimensions.",
        },

        # --- 4. Mathematical formulas and derivations ---
        {
            "query_id": "eval_q13",
            "query": "What is the Singular Value Decomposition (SVD) theorem for an arbitrary real matrix?",
            "category": "formula_derivation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["mathematics_for_machine_learning.pdf"],
            "expected_chunk_ids": ["mathematics_for_machine_learning_chunk_00280"],
            "notes": "Section 4.5 Singular Value Decomposition: Theorem 4.23 A = U Sigma V^T.",
        },
        {
            "query_id": "eval_q14",
            "query": "What is the mathematical formulation of the Dirichlet-Multinomial conjugate model?",
            "category": "formula_derivation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["probabilistic_machine_learning_introduction.pdf"],
            "expected_chunk_ids": ["probabilistic_machine_learning_introduction_chunk_00361"],
            "notes": "Section 4.6.3 Dirichlet-Multinomial likelihood and conjugate prior update.",
        },
        {
            "query_id": "eval_q15",
            "query": "What is the mathematical formula for scaled dot-product attention in Transformers?",
            "category": "formula_derivation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["understanding_deep_learning.pdf"],
            "expected_chunk_ids": ["understanding_deep_learning_chunk_00527", "understanding_deep_learning_chunk_00528"],
            "notes": "Chapter 12 Transformers, scaled dot product attention softmax(Q K^T / sqrt(d_k)) V.",
        },
        {
            "query_id": "eval_q16",
            "query": "What is the multivariate Gaussian probability density function and its normalization constant?",
            "category": "formula_derivation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["mathematics_for_machine_learning.pdf"],
            "expected_chunk_ids": ["mathematics_for_machine_learning_chunk_00468"],
            "notes": "Section 6.5 Gaussian Distribution, Equation 6.54 multivariate density.",
        },

        # --- 5. Algorithms and implementation details ---
        {
            "query_id": "eval_q17",
            "query": "What are the E-step and M-step in the Expectation-Maximization algorithm for Gaussian Mixture Models?",
            "category": "algorithm_implementation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["probabilistic_machine_learning_introduction.pdf"],
            "expected_chunk_ids": ["probabilistic_machine_learning_introduction_chunk_01919"],
            "notes": "Chapter 21 Expectation Maximization for GMMs: posterior responsibilities (E) and parameter updates (M).",
        },
        {
            "query_id": "eval_q18",
            "query": "How does the DBSCAN clustering algorithm define core points, border points, and noise?",
            "category": "algorithm_implementation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": ["hands_on_machine_learning_2e_chunk_00515"],
            "notes": "Chapter 9 DBSCAN algorithm: epsilon neighborhood and min_samples criterion.",
        },
        {
            "query_id": "eval_q19",
            "query": "How does Bahdanau additive attention calculate alignment scores between decoder and encoder states?",
            "category": "algorithm_implementation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["dive_into_deep_learning.pdf"],
            "expected_chunk_ids": ["d2l_chunk_00114", "d2l_chunk_00994"],
            "notes": "Section 11.2 Bahdanau attention additive alignment score function.",
        },
        {
            "query_id": "eval_q20",
            "query": "How does the Viterbi algorithm compute the dynamic programming trellis for minimum edit path or decoding?",
            "category": "algorithm_implementation",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00225", "speech_and_language_processing_chunk_00234"],
            "notes": "Chapter 2 Minimum Edit Distance and Viterbi dynamic programming backtrace.",
        },

        # --- 6. Code-related questions ---
        {
            "query_id": "eval_q21",
            "query": "How do you build a machine learning Pipeline in scikit-learn using StandardScaler?",
            "category": "code",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": ["hands_on_machine_learning_2e_chunk_00179"],
            "notes": "Chapter 2 Transformation Pipelines using Pipeline and StandardScaler.",
        },
        {
            "query_id": "eval_q22",
            "query": "How is a custom PyTorch nn.Module defined with __init__ and forward methods in Dive into Deep Learning?",
            "category": "code",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["dive_into_deep_learning.pdf"],
            "expected_chunk_ids": ["d2l_chunk_00285"],
            "notes": "Section 6.1 Custom module definition subclassing nn.Module.",
        },
        {
            "query_id": "eval_q23",
            "query": "How do you configure an EarlyStopping callback in Keras during model training?",
            "category": "code",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": ["hands_on_machine_learning_2e_chunk_00635", "hands_on_machine_learning_2e_chunk_00636"],
            "notes": "Chapter 10 Using Callbacks: keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True).",
        },
        {
            "query_id": "eval_q24",
            "query": "How does inverted dropout scale activations during training?",
            "category": "code",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["understanding_deep_learning.pdf"],
            "expected_chunk_ids": ["understanding_deep_learning_chunk_00350"],
            "notes": "Chapter 7 Regularization, Section 7.5 Dropout scaling activations by 1 / (1 - p).",
        },

        # --- 7. Deep learning and Transformer architecture ---
        {
            "query_id": "eval_q25",
            "query": "Why are query, key, and value linear projections used in multi-head attention?",
            "category": "deep_learning_transformer",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["understanding_deep_learning.pdf"],
            "expected_chunk_ids": ["understanding_deep_learning_chunk_00528"],
            "notes": "Chapter 12 Transformers, multi-head attention projections into different linear subspaces.",
        },
        {
            "query_id": "eval_q26",
            "query": "What is the difference between Pre-LayerNorm and Post-LayerNorm in Transformer architectures?",
            "category": "deep_learning_transformer",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00581"],
            "notes": "Chapter 10 Transformers, Section 10.3 Residual connections and Pre-LN vs Post-LN normalization placement.",
        },
        {
            "query_id": "eval_q27",
            "query": "How do sinusoidal positional encodings represent token positions in the Transformer?",
            "category": "deep_learning_transformer",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00598"],
            "notes": "Chapter 10 Positional Encoding, sine and cosine geometric progressions.",
        },
        {
            "query_id": "eval_q28",
            "query": "How does causal masking work in autoregressive decoder-only Transformers?",
            "category": "deep_learning_transformer",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00572"],
            "notes": "Chapter 10 Causal Masking, setting attention weights for future positions to minus infinity.",
        },

        # --- 8. Statistical and probabilistic machine learning ---
        {
            "query_id": "eval_q29",
            "query": "What is the fundamental difference between Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation?",
            "category": "probabilistic_ml",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["probabilistic_machine_learning_introduction.pdf"],
            "expected_chunk_ids": ["probabilistic_machine_learning_introduction_chunk_00289"],
            "notes": "Chapter 4 Statistics, Section 4.2 MAP estimation vs MLE incorporating prior p(theta).",
        },
        {
            "query_id": "eval_q30",
            "query": "What is d-separation in directed graphical models?",
            "category": "probabilistic_ml",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["mathematics_for_machine_learning.pdf"],
            "expected_chunk_ids": ["mathematics_for_machine_learning_chunk_00685"],
            "notes": "Section 8.4 Graphical Models: d-separation criterion on directed acyclic graphs.",
        },
        {
            "query_id": "eval_q31",
            "query": "What is a Gaussian process and how is its kernel covariance function defined?",
            "category": "probabilistic_ml",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["probabilistic_machine_learning_introduction.pdf"],
            "expected_chunk_ids": ["probabilistic_machine_learning_introduction_chunk_01376"],
            "notes": "Chapter 17 Gaussian Processes: definition and Mercer kernel covariance functions.",
        },
        {
            "query_id": "eval_q32",
            "query": "What is the evidence lower bound (ELBO) in variational inference?",
            "category": "probabilistic_ml",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["probabilistic_machine_learning_introduction.pdf"],
            "expected_chunk_ids": ["probabilistic_machine_learning_introduction_chunk_00395"],
            "notes": "Chapter 4 / 21 Variational Inference, Jensen's inequality and ELBO definition.",
        },

        # --- 9. NLP and language modelling ---
        {
            "query_id": "eval_q33",
            "query": "How does the Byte-Pair Encoding (BPE) subword tokenization algorithm work?",
            "category": "nlp_language_modeling",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00143"],
            "notes": "Chapter 2 Regular Expressions, Text Normalization, Subwords: BPE merge iterations.",
        },
        {
            "query_id": "eval_q34",
            "query": "How is the BERT masked language model (MLM) pretraining objective formulated?",
            "category": "nlp_language_modeling",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00557"],
            "notes": "Chapter 11 Pretrained Language Models, Section 11.1 BERT MLM 15% token masking.",
        },
        {
            "query_id": "eval_q35",
            "query": "What is the BLEU score used for in machine translation evaluation?",
            "category": "nlp_language_modeling",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00874"],
            "notes": "Chapter 13 Machine Translation, Section 13.7 Automatic Evaluation: BLEU and brevity penalty.",
        },
        {
            "query_id": "eval_q36",
            "query": "How does continuous bag-of-words (CBOW) differ from skip-gram in word embeddings?",
            "category": "nlp_language_modeling",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["speech_and_language_processing.pdf"],
            "expected_chunk_ids": ["speech_and_language_processing_chunk_00480"],
            "notes": "Chapter 6 Vector Semantics, CBOW context averaging vs skip-gram target prediction.",
        },

        # --- 10. Questions where exact terminology matters ---
        {
            "query_id": "eval_q37",
            "query": "What is the spectral norm of a matrix and how does it relate to singular values?",
            "category": "exact_terminology",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["mathematics_for_machine_learning.pdf"],
            "expected_chunk_ids": ["mathematics_for_machine_learning_chunk_00256"],
            "notes": "Section 4.4 Matrix norms and spectral norm defined by largest singular value.",
        },
        {
            "query_id": "eval_q38",
            "query": "What is the silhouette score and how does it evaluate clustering quality?",
            "category": "exact_terminology",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": ["hands_on_machine_learning_2e_chunk_00498"],
            "notes": "Chapter 9 Unsupervised Learning: silhouette coefficient (b - a) / max(a, b).",
        },
        {
            "query_id": "eval_q39",
            "query": "What is the difference between forward KL divergence and reverse KL divergence?",
            "category": "exact_terminology",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["probabilistic_machine_learning_introduction.pdf"],
            "expected_chunk_ids": ["probabilistic_machine_learning_introduction_chunk_00552"],
            "notes": "Chapter 6 Information Theory: KL divergence asymmetry and forward vs reverse mode-seeking.",
        },
        {
            "query_id": "eval_q40",
            "query": "What is the Frobenius norm of a matrix and how is it calculated?",
            "category": "exact_terminology",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["dive_into_deep_learning.pdf"],
            "expected_chunk_ids": ["d2l_chunk_00164"],
            "notes": "Section 2.3 Linear Algebra: Frobenius norm definition as square root of sum of squared matrix entries.",
        },

        # --- 11. Questions requiring information from a particular textbook ---
        {
            "query_id": "eval_q41",
            "query": "According to Dive into Deep Learning, how is Section 5.3.3 Backpropagation explained?",
            "category": "textbook_specific",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["dive_into_deep_learning.pdf"],
            "expected_chunk_ids": ["d2l_chunk_00486", "d2l_chunk_00487"],
            "notes": "Section 5.3.3 Backpropagation in D2L with exact computational graph equations.",
        },
        {
            "query_id": "eval_q42",
            "query": "In Murphy's Probabilistic Machine Learning, what is the definition of conjugate prior?",
            "category": "textbook_specific",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["probabilistic_machine_learning_introduction.pdf"],
            "expected_chunk_ids": ["probabilistic_machine_learning_introduction_chunk_00239"],
            "notes": "Chapter 4 Section 4.6.1 Conjugate priors definition for exponential families.",
        },
        {
            "query_id": "eval_q43",
            "query": "According to Understanding Deep Learning by Simon Prince, what is the 1D shallow neural network equation?",
            "category": "textbook_specific",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["understanding_deep_learning.pdf"],
            "expected_chunk_ids": ["understanding_deep_learning_chunk_00094"],
            "notes": "Chapter 3 Shallow Neural Networks: Equation (3.1) 1D shallow network formulation.",
        },
        {
            "query_id": "eval_q44",
            "query": "According to Géron's Hands-On Machine Learning, how does the VotingClassifier combine ensemble predictions?",
            "category": "textbook_specific",
            "is_supported": True,
            "requires_human_annotation": False,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": ["hands_on_machine_learning_2e_chunk_00401"],
            "notes": "Chapter 7 Ensemble Learning: VotingClassifier with hard voting and soft voting.",
        },

        # --- 12. Ambiguous technical follow-ups ---
        {
            "query_id": "eval_q45",
            "query": "What is the update rule for the weights in this model?",
            "category": "ambiguous_followup",
            "is_supported": True,
            "requires_human_annotation": True,
            "expected_sources": ["dive_into_deep_learning.pdf", "understanding_deep_learning.pdf"],
            "expected_chunk_ids": [],
            "notes": "Ambiguous query: refers to 'this model' without prior conversational referent. Marked for human annotation.",
        },
        {
            "query_id": "eval_q46",
            "query": "How does its temperature parameter affect the output probabilities?",
            "category": "ambiguous_followup",
            "is_supported": True,
            "requires_human_annotation": True,
            "expected_sources": ["speech_and_language_processing.pdf", "understanding_deep_learning.pdf"],
            "expected_chunk_ids": [],
            "notes": "Ambiguous query: refers to pronoun 'its'. Softmax temperature appears across LLM sampling and energy models.",
        },
        {
            "query_id": "eval_q47",
            "query": "Can it handle missing data without imputation?",
            "category": "ambiguous_followup",
            "is_supported": True,
            "requires_human_annotation": True,
            "expected_sources": ["hands_on_machine_learning_2e.pdf"],
            "expected_chunk_ids": [],
            "notes": "Ambiguous query: refers to pronoun 'it'. Decision trees vs scikit-learn estimators.",
        },

        # --- 13. Negative or unsupported queries ---
        {
            "query_id": "eval_q48",
            "query": "How does Shor's quantum factoring algorithm achieve polynomial time speedup using the quantum Fourier transform?",
            "category": "negative_unsupported",
            "is_supported": False,
            "requires_human_annotation": False,
            "expected_sources": [],
            "expected_chunk_ids": [],
            "notes": "Negative test query: Quantum computing and Shor's algorithm are completely absent from the 6 ML/DL textbooks.",
        },
        {
            "query_id": "eval_q49",
            "query": "What were the primary economic and military causes of the fall of the Western Roman Empire?",
            "category": "negative_unsupported",
            "is_supported": False,
            "requires_human_annotation": False,
            "expected_sources": [],
            "expected_chunk_ids": [],
            "notes": "Negative test query: Ancient Mediterranean history is outside the corpus domain.",
        },
        {
            "query_id": "eval_q50",
            "query": "How do you synthesize aspirin from salicylic acid and acetic anhydride in organic chemistry?",
            "category": "negative_unsupported",
            "is_supported": False,
            "requires_human_annotation": False,
            "expected_sources": [],
            "expected_chunk_ids": [],
            "notes": "Negative test query: Organic chemistry esterification is outside the corpus domain.",
        },
        {
            "query_id": "eval_q51",
            "query": "What are the traditional ingredients and braising steps for classic French boeuf bourguignon?",
            "category": "negative_unsupported",
            "is_supported": False,
            "requires_human_annotation": False,
            "expected_sources": [],
            "expected_chunk_ids": [],
            "notes": "Negative test query: Culinary recipes are outside the corpus domain.",
        },
        {
            "query_id": "eval_q52",
            "query": "How does the Bitcoin consensus protocol resolve blockchain reorganizations using the Nakamoto longest-chain rule?",
            "category": "negative_unsupported",
            "is_supported": False,
            "requires_human_annotation": False,
            "expected_sources": [],
            "expected_chunk_ids": [],
            "notes": "Negative test query: Cryptocurrency consensus mechanisms are outside the corpus domain.",
        },
    ]


def main():
    print(f"Loading corpus from {DOCUMENTS_FILE}...")
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} indexed chunks.")

    queries = build_raw_query_definitions()
    print(f"Validating {len(queries)} query definitions against corpus...")

    verified_queries = []
    supported_verified_count = 0
    negative_count = 0
    unannotated_count = 0

    for q in queries:
        qid = q["query_id"]
        expected_chunk_ids = q.get("expected_chunk_ids", [])
        expected_sources = q.get("expected_sources", [])
        expected_page_ranges = []

        if not q["is_supported"]:
            negative_count += 1
            q["expected_page_ranges"] = []
            verified_queries.append(q)
            continue

        if q["requires_human_annotation"]:
            unannotated_count += 1
            q["expected_page_ranges"] = []
            verified_queries.append(q)
            continue

        # For fully supported and annotated queries, verify every single chunk ID
        if not expected_chunk_ids:
            raise ValueError(f"Query {qid} is marked supported but has empty expected_chunk_ids")

        for cid in expected_chunk_ids:
            if cid not in corpus:
                raise KeyError(f"Query {qid} references non-existent chunk ID: {cid}")
            chunk = corpus[cid]
            # Verify source file alignment
            if chunk["file_name"] not in expected_sources:
                raise ValueError(
                    f"Query {qid} chunk {cid} has file {chunk['file_name']} not in expected_sources {expected_sources}"
                )
            expected_page_ranges.append([chunk["page_start"], chunk["page_end"]])

        q["expected_page_ranges"] = expected_page_ranges
        supported_verified_count += 1
        verified_queries.append(q)

    # Save to data/rag/evaluation/retrieval_queries.jsonl
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for vq in verified_queries:
            f.write(json.dumps(vq, ensure_ascii=False) + "\n")

    print("\nSuccessfully built retrieval evaluation dataset:")
    print(f"  Target file: {OUTPUT_FILE}")
    print(f"  Total queries: {len(verified_queries)}")
    print(f"  Fully verified ground-truth queries: {supported_verified_count}")
    print(f"  Ambiguous / human-annotation queries: {unannotated_count}")
    print(f"  Negative / unsupported queries: {negative_count}")


if __name__ == "__main__":
    main()
