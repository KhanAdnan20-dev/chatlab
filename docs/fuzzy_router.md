# Soft Computing: Mamdani Fuzzy Logic Decision Router

This document provides the mathematical, theoretical, and engineering specification of the **Mamdani Fuzzy Logic Decision Router** implemented in ChatLab.

---

## 1. Motivation: Why Soft Computing & Fuzzy Logic?

### 1.1 The Limitation of Crisp Thresholding
In conversational AI architectures, routing decisions between deterministic NLP layers and generative LLM models are inherently nuanced. Simple crisp rules (e.g., `if confidence > 0.6: Layer1 else: Layer2`) suffer from severe boundary instabilities:
- A query with confidence $0.59$ versus $0.61$ triggers radically different execution pathways despite negligible semantic difference.
- Crisp logic fails to capture the interdependency between multiple uncertainty signals—such as high intent confidence on a short greeting versus high confidence on a complex, multi-clause question.

### 1.2 The Fuzzy Logic Solution
Fuzzy Logic (Zadeh, 1965) maps crisp numerical observations into continuous degrees of truth $\mu \in [0, 1]$. By evaluating multi-criteria linguistic rules, the system makes smooth, interpretable, and robust routing decisions.

---

## 2. Router Architecture Diagram

```
                              Incoming User Query
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       [Intent Classifier]                           [Entity Extractor]
     Intent Confidence C ∈ [0, 1]                  Entity Density E ∈ [0, 1]
                │                                             │
                ▼                                             ▼
       [Semantic Similarity]                         [Lexical Analyzer]
     Similarity Score S ∈ [0, 1]                   Query Complexity L ∈ [0, 1]
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │        FUZZIFICATION          │
                       │ (Triangular & Trapezoidal MFs)│
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     MAMDANI RULE EVALUATION   │
                       │    (18 Rules, Min T-norm)     │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     MAX AGGREGATION &         │
                       │   CENTROID DEFUZZIFICATION    │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       Layer Activation Score A ∈ [0, 1]
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
             A <= 0.50                                  A > 0.50
                   │                                       │
                   ▼                                       ▼
         [Dispatch Layer 1]                     [Route to Layer 2]
    Deterministic / Curated NLP              Generative LLM (Pending)
```

---

## 3. Fuzzy Input Variables

The router evaluates four crisp input variables normalized into $[0, 1]$:

### 3.1 Intent Confidence ($C \in [0, 1]$)
The true posterior probability $P(\text{intent} \mid \mathbf{x})$ output by the Layer 1 multi-class Logistic Regression classifier.
- **LOW**: $\text{trapmf}(C, [0.0, 0.0, 0.25, 0.45])$
- **MEDIUM**: $\text{trimf}(C, [0.30, 0.55, 0.75])$
- **HIGH**: $\text{trapmf}(C, [0.60, 0.80, 1.0, 1.0])$

### 3.2 Semantic Similarity ($S \in [0, 1]$)
The cosine similarity against the closest training pattern in vector space:
$$S = \max_{p \in \mathcal{P}} \frac{\mathbf{q} \cdot \mathbf{p}}{\|\mathbf{q}\|_2 \|\mathbf{p}\|_2}$$
- **LOW**: $\text{trapmf}(S, [0.0, 0.0, 0.20, 0.40])$
- **MEDIUM**: $\text{trimf}(S, [0.25, 0.50, 0.70])$
- **HIGH**: $\text{trapmf}(S, [0.55, 0.75, 1.0, 1.0])$

### 3.3 Query Complexity ($L \in [0, 1]$)
Lexical complexity combining token count ($N_{\text{tok}}$), character count ($N_{\text{char}}$), and lexical diversity ($D_{\text{lex}} = \frac{|\text{unique tokens}|}{N_{\text{tok}}}$):
$$L = \text{clip}\left(0.50 \cdot \min\left(\frac{N_{\text{tok}}}{18}, 1\right) + 0.35 \cdot \min\left(\frac{N_{\text{char}}}{100}, 1\right) + 0.15 \cdot D_{\text{lex}}, 0, 1\right)$$
- **SIMPLE**: $\text{trapmf}(L, [0.0, 0.0, 0.20, 0.40])$
- **MODERATE**: $\text{trimf}(L, [0.25, 0.50, 0.70])$
- **COMPLEX**: $\text{trapmf}(L, [0.55, 0.75, 1.0, 1.0])$

### 3.4 Entity Density ($E \in [0, 1]$)
Ratio of recognized named entities ($N_{\text{ent}}$) to total tokens:
$$E = \min\left(\frac{N_{\text{ent}}}{\max(N_{\text{tok}}, 1)} \cdot 2.2, 1.0\right)$$
- **LOW**: $\text{trapmf}(E, [0.0, 0.0, 0.15, 0.35])$
- **MEDIUM**: $\text{trimf}(E, [0.20, 0.45, 0.70])$
- **HIGH**: $\text{trapmf}(E, [0.50, 0.75, 1.0, 1.0])$

---

## 4. Fuzzy Output Variable: Layer Activation Score ($A \in [0, 1]$)

The output universe $Z = [0, 1]$ models the activation propensity toward Layer 2:
- **LAYER1** (Low Activation &rarr; Deterministic NLP): $\text{trapmf}(z, [0.0, 0.0, 0.20, 0.40])$
- **HYBRID** (Uncertain / Mixed Region): $\text{trimf}(z, [0.25, 0.50, 0.75])$
- **LAYER2** (High Activation &rarr; Generative LLM): $\text{trapmf}(z, [0.60, 0.80, 1.0, 1.0])$

---

## 5. Fuzzy Rule Base (18 Mamdani Rules)

| Rule | Antecedent Conditions (IF) | Consequent (THEN) | Academic Rationale |
| :--- | :--- | :--- | :--- |
| **R1** | $C$ is HIGH $\land$ $S$ is HIGH $\land$ $L$ is SIMPLE | Activation is **LAYER1** | Clear standard greeting or FAQ query. |
| **R2** | $C$ is HIGH $\land$ $S$ is HIGH $\land$ $L$ is MODERATE | Activation is **LAYER1** | Known intent with moderate phrasing length. |
| **R3** | $C$ is LOW $\land$ $S$ is LOW $\land$ $L$ is COMPLEX | Activation is **LAYER2** | Unseen open-ended or difficult query. |
| **R4** | $C$ is LOW $\land$ $S$ is MEDIUM $\land$ $L$ is COMPLEX | Activation is **LAYER2** | Vaguely related phrasing requiring LLM reasoning. |
| **R5** | $C$ is MEDIUM $\land$ $S$ is MEDIUM $\land$ $L$ is MODERATE | Activation is **HYBRID** | Boundary case between NLP and generative. |
| **R6** | $C$ is HIGH $\land$ $L$ is COMPLEX | Activation is **HYBRID** | High confidence but complex syntax. |
| **R7** | $C$ is LOW $\land$ $S$ is HIGH $\land$ $L$ is SIMPLE | Activation is **LAYER1** | Semantic similarity rescue for short queries. |
| **R8** | $C$ is HIGH $\land$ $E$ is HIGH $\land$ $L$ is SIMPLE | Activation is **LAYER1** | High confidence with clear entity recognition. |
| **R9** | $C$ is MEDIUM $\land$ $S$ is HIGH $\land$ $L$ is SIMPLE | Activation is **LAYER1** | Rescued simple query with strong pattern match. |
| **R10**| $C$ is LOW $\land$ $S$ is LOW | Activation is **LAYER2** | Out-of-domain unhandled query. |
| **R11**| $C$ is LOW $\land$ $S$ is MEDIUM $\land$ $E$ is LOW | Activation is **LAYER2** | Low confidence without entity grounding. |
| **R12**| $C$ is HIGH $\land$ $S$ is LOW $\land$ $L$ is SIMPLE | Activation is **LAYER1** | Novel phrasing of known intent. |
| **R13**| $C$ is MEDIUM $\land$ $S$ is HIGH $\land$ $L$ is MODERATE | Activation is **LAYER1** | High similarity pattern match. |
| **R14**| $L$ is MODERATE $\land$ $E$ is HIGH $\land$ $C$ is MEDIUM | Activation is **HYBRID** | Rich entity query requiring careful arbitration. |
| **R15**| $L$ is COMPLEX $\land$ $E$ is HIGH $\land$ $C$ is LOW | Activation is **LAYER2** | Multi-entity complex reasoning query. |
| **R16**| $C$ is HIGH $\land$ $S$ is MEDIUM $\land$ $L$ is SIMPLE | Activation is **LAYER1** | Confident simple response. |
| **R17**| $C$ is MEDIUM $\land$ $S$ is LOW $\land$ $L$ is MODERATE | Activation is **HYBRID** | Moderate uncertainty across features. |
| **R18**| $L$ is COMPLEX $\land$ $S$ is MEDIUM | Activation is **LAYER2** | Long complex sentence requiring generative handling. |

---

## 6. Fuzzy Inference & Defuzzification

1. **Fuzzy T-norm (AND)**: Minimum operator:
   $$\alpha_r = \min(\mu_{C, r}(C), \mu_{S, r}(S), \mu_{L, r}(L), \mu_{E, r}(E))$$
2. **Implication**: Mamdani minimum clipping:
   $$\mu_{\text{implied}, r}(z) = \min(\alpha_r, \mu_{\text{out}, r}(z))$$
3. **Aggregation**: Maximum operator across all 18 rules:
   $$\mu_{\text{agg}}(z) = \max_{r=1}^{18} \mu_{\text{implied}, r}(z)$$
4. **Centroid Defuzzification (Center of Gravity)**:
   $$A^* = \frac{\int_{0}^{1} z \cdot \mu_{\text{agg}}(z) \, dz}{\int_{0}^{1} \mu_{\text{agg}}(z) \, dz}$$

---

## 7. Decision Mechanism

$$\text{Selected Layer} = \begin{cases} \text{Layer 1 (Deterministic NLP)}, & \text{if } A^* \le 0.50 \text{ and intent} \neq \text{"unknown"} \\ \text{Layer 2 (Generative LLM Pending)}, & \text{if } A^* > 0.50 \text{ or intent} = \text{"unknown"} \end{cases}$$
