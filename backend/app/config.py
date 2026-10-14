import os
from pathlib import Path
from pydantic import BaseModel

# Project root derived from this file's location
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    """Centralized application configuration settings for ChatLab."""

    PROJECT_NAME: str = "ChatLab"
    VERSION: str = "0.3.0"
    DESCRIPTION: str = "ChatLab Modular AIML QA and Quiz Chatbot"
    BASE_DIR: Path = BASE_DIR

    # Dataset paths
    INTENT_DATA_PATH: Path = BASE_DIR / "data" / "intents.json"
    TOPICS_DATA_PATH: Path = BASE_DIR / "data" / "aiml_topics.json"
    QUIZ_DATA_PATH: Path = BASE_DIR / "data" / "quiz_questions.json"
    QA_DATASET_PATH: Path = BASE_DIR / "data" / "qa_dataset.jsonl"
    QA_TRAIN_PATH: Path = BASE_DIR / "data" / "qa_train.jsonl"
    QA_VALIDATION_PATH: Path = BASE_DIR / "data" / "qa_validation.jsonl"
    QA_TEST_PATH: Path = BASE_DIR / "data" / "qa_test.jsonl"

    # Model and artifacts paths
    MODELS_DIR: Path = BASE_DIR / "models"
    INTENT_MODEL_PATH: Path = BASE_DIR / "models" / "intent_classifier.joblib"
    EMBEDDING_MODEL_PATH: Path = BASE_DIR / "models" / "all-MiniLM-L6-v2"

    # RAG Retrieval Settings
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
    RAG_INDEX_DIR: Path = BASE_DIR / "data" / "rag" / "indexes" / "faiss"
    RAG_DOCUMENTS_PATH: Path = BASE_DIR / "data" / "rag" / "processed" / "documents.jsonl"
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
    RAG_MIN_SCORE: float = float(os.getenv("RAG_MIN_SCORE", "0.35"))

    # Gemma Model Settings
    GEMMA_MODEL_ID: str = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
    USE_4BIT_QUANTIZATION: bool = os.getenv("USE_4BIT_QUANTIZATION", "true").lower() == "true"
    DEVICE_MAP: str = os.getenv("DEVICE_MAP", "auto")
    GEMMA_ENABLED: bool = True
    TOKEN_BUDGET_SIMPLE: int = 96
    TOKEN_BUDGET_NORMAL: int = 192
    TOKEN_BUDGET_COMPLEX: int = 384

    # DeepSeek-R1 Distill Reasoning Engine Settings
    DEEPSEEK_MODEL_ID: str = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    DEEPSEEK_ENABLED: bool = os.getenv("DEEPSEEK_ENABLED", "true").lower() == "true"
    DEFAULT_MODEL_CHOICE: str = os.getenv("DEFAULT_MODEL_CHOICE", "gemma")


    # vLLM High-Throughput Serving Engine Configuration
    VLLM_ENABLED: bool = os.getenv("VLLM_ENABLED", "true").lower() == "true"
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "google/gemma-4-E2B-it")

    # Operational flags
    OFFLINE_MODE: bool = False
    LAYER2_ENABLED: bool = True

    # NLP thresholds
    LAYER1_CONFIDENCE_THRESHOLD: float = 0.55
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.40
    FALLBACK_RESPONSE: str = (
        "I understand the question, but this request requires the advanced "
        "conversational model (Layer 2), which is not enabled yet."
    )

    # Conversation memory
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "20"))

    # Quiz defaults
    QUIZ_DEFAULT_DIFFICULTY: str = "medium"
    QUIZ_DEFAULT_QUESTION_COUNT: int = 5
    QUIZ_MAX_QUESTION_COUNT: int = 20

    # CORS configuration for frontend development
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


settings = Settings()
