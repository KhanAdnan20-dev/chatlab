# ChatLab REST API Specification

## Endpoints

### 1. Root Status
- **`GET /`**
- **Response**: `RootResponse` (`message`, `version`, `docs_url`)

### 2. Health Check
- **`GET /health`**
- **Response**: `HealthResponse` (`status`, `model_loaded`, `intents_count`, `layer2_enabled`, `active_sessions_count`)

### 3. Send Chat Message
- **`POST /chat`**
- **Request Body**:
  ```json
  {
    "message": "What is Supervised Learning?",
    "session_id": "sess-1234"
  }
  ```
- **Response**: `ChatResponse` (includes `response`, `layer`, `intent`, `confidence`, `activation_score`, `routing_reason`, `processing_time_ms`, and `citations`).
- **Citation Metadata Structure (`citations: list[CitationItem]`)**:
  ```json
  [
    {
      "citation_code": "[S1]",
      "source_title": "Understanding Deep Learning",
      "section": "PDF p. 229",
      "content_snippet": "12.4 Transformer layers Figure 12.6 Multi-head attention...",
      "formulas": [],
      "file_name": "understanding_deep_learning.pdf",
      "page_start": 229,
      "page_end": 229,
      "subject": "deep_learning",
      "score": 0.6961,
      "source_url": "https://udlbook.github.io/udlbook/"
    }
  ]
  ```
  *Note: If retrieval is bypassed or returns no results above the similarity threshold, `citations` is an empty list `[]`.*

### 4. Start Quiz
- **`POST /quiz/start`**
- **Request Body**:
  ```json
  {
    "topic": "all",
    "difficulty": "medium",
    "num_questions": 5
  }
  ```

### 5. Submit Quiz Answer
- **`POST /quiz/{session_id}/answer`**
- **Request Body**:
  ```json
  {
    "answer_index": 1
  }
  ```

### 6. Advance Question
- **`POST /quiz/{session_id}/next`**

### 7. Get Quiz Score
- **`GET /quiz/{session_id}/score`**

### 8. Finish Quiz
- **`POST /quiz/{session_id}/finish`**
