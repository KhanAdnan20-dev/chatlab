# ChatLab — A Hybrid NLP Conversational AI System

ChatLab is an academic conversational AI system designed to demonstrate a **modular, multi-layer hybrid architecture**. 

Incoming queries pass through text preprocessing, an Advanced NLP Layer (Layer 1), and a **Soft-Computing Mamdani Fuzzy Inference Router** that arbitrates between deterministic Layer 1 responses and routing complex generative queries to an open-source LLM (Layer 2).

> **Current Implementation Phase:** Complete implementation of **Layer 1 (Advanced NLP Foundation)** and the **Mamdani Fuzzy Logic Decision Router (Phase 2)**, featuring multi-criteria fuzzification, 18-rule fuzzy inference, centroid defuzzification, FastAPI REST backend, React + Vite frontend with a live **Soft Computing & NLP Inspector**, and automated test suite. *(Layer 2 local LLM will be integrated in subsequent phases).*

---

## 🏛️ System Architecture

```
                                  USER
                                    │
                                    ▼
                              Preprocessing
                       (Normalization, Tokens, Lemmas)
                                    │
                                    ▼
                            Advanced NLP Layer
                    ┌───────────────────────────────┐
                    │ • TF-IDF + Logistic Classifier│
                    │ • Statistical & Gazetteer NER │
                    │ • Cosine Semantic Similarity  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                     [MAMDANI FUZZY LOGIC ROUTER]
            (Confidence C, Similarity S, Complexity L, Entity Density E)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
          Activation <= 0.50                  Activation > 0.50
                  │                                   │
                  ▼                                   ▼
          ┌───────────────┐                   ┌───────────────┐
          │    Layer 1    │                   │    Layer 2    │
          │ Advanced NLP  │                   │ Fine-Tuned LLM│
          │ (Active Now)  │                   │ (Pending LLM) │
          └───────┬───────┘                   └───────────────┘
                  │
                  ▼
          Final Response + Full Soft-Computing & NLP Metadata
```

---

## ✨ Features (Layer 1 + Phase 2 Fuzzy Router)

- **Mamdani Fuzzy Inference System (Soft Computing)**:
  - 4 Fuzzy Input Variables: Intent Confidence ($C$), Semantic Similarity ($S$), Query Complexity ($L$), Entity Density ($E$).
  - Triangular & Trapezoidal Membership Functions (`trimf`, `trapmf`).
  - 18 Academic Mamdani Rules evaluating multi-criteria uncertainty.
  - Centroid (Center of Gravity) Defuzzification producing continuous Layer Activation Score $A \in [0, 1]$.
- **Text Preprocessing Pipeline**: Unicode NFKD normalization, regex-based tokenization, whitespace cleanup, and WordNet lemmatization.
- **Machine Learning Intent Classifier**: Multi-class Logistic Regression with FeatureUnion (word n-grams + character n-grams) trained on 17 intent categories.
- **Semantic Similarity Engine**: Vector space pattern matching using sublinear TF-IDF and cosine similarity for nearest-neighbor rescue.
- **Named Entity Recognition (NER)**: Hybrid entity extraction combining NLTK statistical POS chunking and domain gazetteers.
- **Interactive Modern Web Interface**: React + Vite single-page application featuring a real-time **Soft Computing & NLP Inspector** displaying confidence meters, fuzzy inputs, activation score, and routing reason.
- **RESTful API**: FastAPI endpoints (`GET /`, `GET /health`, `POST /chat`) with strict Pydantic v2 schemas and CORS support.

---

## 🛠️ Technology Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Soft Computing / Fuzzy Logic** | Mamdani FIS, NumPy | Continuous fuzzy inference and centroid defuzzification |
| **Backend Framework** | FastAPI, Uvicorn, Starlette | High-performance async REST API |
| **Data Validation** | Pydantic v2, Pydantic Settings | Request/response data validation |
| **NLP & Machine Learning** | Scikit-Learn, NLTK, NumPy, SciPy | TF-IDF, Logistic Regression, POS/NER |
| **Model Serialization** | Joblib | Pretrained model artifact persistence |
| **Frontend Framework** | React 18, Vite | Interactive single-page chat interface |
| **Testing** | Pytest, HTTPX | Unit and end-to-end integration tests |

---

## 📁 Project Structure

```
ChatLab/
├── backend/
│   ├── main.py                      # FastAPI app entry point & CORS configuration
│   └── app/
│       ├── __init__.py
│       ├── config.py                # Centralized project configuration
│       ├── schemas.py               # Pydantic request/response schemas
│       ├── router.py                # API endpoints (GET /, GET /health, POST /chat)
│       └── services/
│           ├── __init__.py
│           ├── preprocessing.py     # Text cleaning, tokenization, lemmatization
│           ├── intent_classifier.py # ML intent prediction with true probabilities
│           ├── entity_extractor.py  # Statistical and gazetteer NER
│           ├── semantic_similarity.py# Cosine similarity pattern matching
│           ├── fuzzy_router.py      # Mamdani Fuzzy Inference System (Soft Computing)
│           ├── response_engine.py   # Layer 1 template response dispatcher
│           └── chatbot.py           # Core NLP & Fuzzy pipeline orchestrator
├── data/
│   └── intents.json                 # 17-class dataset with 256 patterns & responses
├── models/
│   └── intent_classifier.joblib     # Pretrained ML model artifact
├── frontend/
│   ├── index.html                   # Frontend HTML shell
│   ├── package.json                 # Frontend dependencies & scripts
│   ├── vite.config.js               # Vite development server configuration
│   └── src/
│       ├── main.jsx                 # React root mount
│       ├── App.jsx                  # Main chat application & state management
│       ├── index.css                # Modern CSS design system
│       └── components/
│           ├── MessageItem.jsx      # Message bubble component
│           └── NlpInspector.jsx     # Soft Computing & NLP parameter inspector panel
├── scripts/
│   ├── train_intent_model.py        # Model training and cross-validation evaluation script
│   └── test_router_examples.py     # Fuzzy router experiment and demonstration script
├── tests/
│   ├── test_preprocessing.py        # Unit tests for preprocessing
│   ├── test_intent_classifier.py    # Unit tests for intent classifier
│   ├── test_semantic_similarity.py  # Unit tests for semantic similarity
│   ├── test_entity_extractor.py     # Unit tests for entity extraction
│   ├── test_fuzzy_router.py         # Unit tests for Mamdani fuzzy router
│   └── test_api.py                  # Integration tests for FastAPI endpoints
├── docs/
│   ├── architecture.md              # In-depth system architecture & flowcharts
│   ├── setup.md                     # Step-by-step setup and running instructions
│   ├── layer1.md                    # Academic formulation of Layer 1 NLP
│   └── fuzzy_router.md              # Mathematical formulation of Mamdani Fuzzy Router
├── requirements.txt                 # Direct Python project dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Retraining the Layer 1 Model
```powershell
python scripts/train_intent_model.py
```

### 3. Evaluating Fuzzy Router Experiments
```powershell
python scripts/test_router_examples.py
```

### 4. Starting the Backend API
```powershell
uvicorn backend.main:app --reload
```
Interactive documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Starting the Frontend UI
```powershell
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 6. Running Automated Tests
```powershell
python -m pytest -v
```

---

## 📡 API Endpoints

### `POST /chat`
Accepts user message and returns response with full Soft Computing and NLP diagnostics:

**Request:**
```json
{
  "message": "I have a Python exam on Monday and need study tips"
}
```

**Response:**
```json
{
  "response": "To study effectively: break large subjects into modular chunks...",
  "layer": "layer1",
  "intent": "study_help",
  "confidence": 0.86,
  "entities": [
    { "text": "Python", "label": "PROGRAMMING_LANGUAGE" },
    { "text": "exam", "label": "ACADEMIC_TERM" },
    { "text": "Monday", "label": "DATE" }
  ],
  "similarity_score": 0.62,
  "matched_pattern": "how should i study for exams",
  "complexity_score": 0.40,
  "entity_density": 0.37,
  "activation_score": 0.18,
  "routing_reason": "Fuzzy activation score (0.18 <= 0.50) confirms Layer 1 suitability.",
  "fallback": false,
  "processing_time_ms": 12.4
}
```
