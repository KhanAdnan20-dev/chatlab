# ChatLab Service Architecture Reference

## 1. Chat Orchestrator (`backend/app/services/chatbot.py`)
- **Responsibility**: Central coordinator for end-to-end NLP and Fuzzy Routing workflow.
- **Inputs**: `message: str`, `session_id: Optional[str]`
- **Outputs**: `ChatbotResult` dataclass
- **Dependencies**: `intent_classifier`, `entity_extractor`, `semantic_similarity`, `fuzzy_router`, `chat_engine`, `conversation_memory`, `response_postprocessor`.

## 2. Model Manager (`backend/app/services/model_manager.py`)
- **Responsibility**: Centralized singleton lifecycle manager for ML models. Implements lazy-loading for GPU models to prevent VRAM allocation during fine-tuning runs.

## 3. Chat Engine Abstraction (`backend/app/services/chat_engine.py`)
- **Responsibility**: Provides standardized `generate(request) -> EngineResponse` interface for Layer 1 and Layer 2.

## 4. Conversation Memory (`backend/app/services/conversation_memory.py`)
- **Responsibility**: In-memory session manager keeping a rolling history window of user and assistant messages for multi-turn context.

## 5. Response Postprocessor (`backend/app/services/response_postprocessor.py`)
- **Responsibility**: Validates, trims, and cleans output text before returning it to the user.

## 6. Quiz Engine (`backend/app/services/quiz_engine.py`)
- **Responsibility**: In-memory state machine for quiz initialization, option evaluation, scoring, and progress tracking. Independent from the Chat service.
