# ChatLab Setup & Installation Guide

This guide walks through setting up the environment, installing dependencies, training Layer 1 models, and running both backend and frontend servers.

---

## 1. Prerequisites
- **Python**: 3.10+ (tested on Python 3.14)
- **Node.js**: 18+ (tested on Node.js 24)
- **npm**: 9+

---

## 2. Python Virtual Environment Setup

From the `ChatLab/` root directory:

```powershell
# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip and install Layer 1 dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Training the Layer 1 Intent Classifier

Before starting the server, train and serialize the machine learning intent model:

```powershell
python scripts/train_intent_model.py
```

This will:
- Load the 17-class dataset from `data/intents.json`.
- Execute 5-fold stratified cross-validation and print evaluation metrics.
- Save the trained pipeline artifact to `models/intent_classifier.joblib`.

---

## 4. Running the FastAPI Backend Server

Launch the FastAPI backend with hot-reload enabled:

```powershell
uvicorn backend.main:app --reload
```

- **API Base URL**: `http://127.0.0.1:8000`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

---

## 5. Running the React + Vite Frontend

In a separate terminal window:

```powershell
cd frontend
npm install
npm run dev
```

- **Frontend URL**: `http://localhost:5173`

---

## 6. Running the Automated Test Suite

Execute the complete test suite (unit tests and integration tests) using `pytest`:

```powershell
python -m pytest -v
```
