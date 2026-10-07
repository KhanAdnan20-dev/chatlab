# Layer 1 NLP Pipeline — Academic & Technical Reference

This document outlines the theoretical foundations, mathematical representations, and implementation specifics of **Layer 1** in the ChatLab conversational system.

---

## 1. Pipeline Overview

Layer 1 operates as a deterministic and statistical Natural Language Understanding (NLU) engine. It is designed to handle common conversational domains with high precision, minimal latency (< 20ms), and full interpretability.

---

## 2. Text Preprocessing

### 2.1 Unicode Normalization & Cleaning
Input string $T_{\text{raw}}$ is normalized using the NFKD decomposition standard:
$$T_{\text{norm}} = \text{strip}(\text{lowercase}(\text{NFKD}(T_{\text{raw}})))$$

### 2.2 Tokenization & Lemmatization
The string is segmented into word tokens $\mathcal{W} = \{w_1, w_2, \dots, w_n\}$ using regular expression boundary matching. Morphological variations are reduced to canonical dictionary lemmas using the WordNet database:
$$l_i = \text{Lemmatize}(w_i, \text{pos})$$

---

## 3. Supervised Intent Classification

### 3.1 Feature Extraction: Dual-Space TF-IDF
To handle both vocabulary variations and spelling irregularities/contractions, features are extracted via a `FeatureUnion`:
1. **Word N-Grams**: Unigrams, bigrams, and trigrams ($n \in [1, 3]$) capturing syntactic phrases.
2. **Character Boundary N-Grams**: Sub-word character sequences ($n \in [2, 5]$) capturing morphology and root patterns.

Term Frequency-Inverse Document Frequency (TF-IDF) with sublinear term scaling is computed as:
$$\text{TF-IDF}(t, d, D) = (1 + \log(\text{tf}(t, d))) \cdot \left(\log\frac{1 + |D|}{1 + \text{df}(t, D)} + 1\right)$$

### 3.2 Classification Model
A multi-class **Logistic Regression** classifier with balanced class weighting estimates posterior class probabilities:
$$P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$
where $\mathbf{x}$ is the combined sparse feature vector and $K = 17$ distinct intent categories.

---

## 4. Semantic Similarity & Pattern Matching

To detect semantic relatedness when exact words differ, queries are projected into the vector space $\mathbb{R}^V$ of all known training patterns $\{\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_M\}$. The cosine similarity is evaluated:
$$\text{CosineSimilarity}(\mathbf{q}, \mathbf{p}_i) = \frac{\mathbf{q} \cdot \mathbf{p}_i}{\|\mathbf{q}\|_2 \|\mathbf{p}_i\|_2}$$
The highest scoring pattern provides both a similarity metric $S \in [0, 1]$ and a candidate intent to rescue borderline classifier predictions.

---

## 5. Named Entity Recognition (NER)

Entity extraction combines two complementary approaches:
1. **Statistical Chunking**: Uses NLTK Part-of-Speech tagging with Maximum Entropy shallow parsing (`ne_chunk`) to detect proper nouns and location chunks.
2. **Domain-Specific Gazetteer**: Exact regex boundary matching against technical ontologies (programming languages, frameworks, AI concepts, exam terms, calendar days).

---

## 6. Evaluation & Benchmarking

Layer 1 performance was measured using **Stratified 5-Fold Cross-Validation** over 256 annotated utterances across 17 intent classes:
- **Mean Accuracy**: ~64.5%
- **Macro F1-Score**: 0.64
- **Weighted F1-Score**: 0.64
- **Average Latency**: 5 to 15 ms per query
