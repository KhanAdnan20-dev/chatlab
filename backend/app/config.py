from pathlib import Path
from pydantic import BaseModel, Field

# Absolute path to the ChatLab project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    """Centralized application configuration settings for ChatLab."""

    PROJECT_NAME: str = "ChatLab"
    VERSION: str = "0.2.0"
    DESCRIPTION: str = (
        "ChatLab - Modular AIML Question-Answering & Quiz Chatbot (Layer 1 + Fuzzy Router + Layer 2 Prep)"
    )

    # Dataset paths
    INTENT_DATA_PATH: Path = BASE_DIR / "data" / "intents.json"
    TOPICS_DATA_PATH: Path = BASE_DIR / "data" / "aiml_topics.json"
    QA_DATASET_PATH: Path = BASE_DIR / "data" / "qa_dataset.jsonl"
    QA_TRAIN_PATH: Path = BASE_DIR / "data" / "qa_train.jsonl"
    QA_VALIDATION_PATH: Path = BASE_DIR / "data" / "qa_validation.jsonl"
    QA_TEST_PATH: Path = BASE_DIR / "data" / "qa_test.jsonl"

    # Model and artifacts paths
    MODELS_DIR: Path = BASE_DIR / "models"
    INTENT_MODEL_PATH: Path = BASE_DIR / "models" / "intent_classifier.joblib"
    EMBEDDING_MODEL_PATH: Path = BASE_DIR / "models" / "all-MiniLM-L6-v2"
    QWEN_MODEL_PATH: Path = BASE_DIR / "models" / "qwen3-4b-bnb-4bit"
    QWEN_ADAPTER_OUTPUT_DIR: Path = BASE_DIR / "models" / "qwen3-4b-qlora-adapter"

    # Offline mode flag
    OFFLINE_MODE: bool = True

    # NLP thresholds and parameters
    LAYER1_CONFIDENCE_THRESHOLD: float = 0.55
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.40
    FALLBACK_RESPONSE: str = (
        "I understand the question, but this request requires the advanced conversational "
        "model (Layer 2), which is not enabled yet."
    )

    # CORS configuration for frontend development
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ]


settings = Settings()
