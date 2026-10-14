"""Pydantic schemas for ChatLab Chat and Quiz APIs."""
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Schema for incoming user chat requests."""

    message: str = Field(
        ...,
        min_length=1,
        description="Natural-language input text message from the user",
        examples=["What is Supervised Learning?", "Teach me recursion"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for multi-turn history tracking",
        examples=["sess-1234"],
    )
    socratic_mode: bool = Field(
        default=False,
        description="Whether to enable Socratic active-recall tutoring mode instead of direct exposition",
    )
    depth_mode: str = Field(
        default="auto",
        description="Pedagogical response depth mode: 'auto' (Mamdani FIS), 'easy', 'medium', 'hard', or 'ultra'",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-generated UUID/request identifier to deduplicate stream retries",
    )
    model_choice: str = Field(
        default="gemma",
        description="Active model engine selection: 'gemma' (Tutor) or 'deepseek' (Reasoning)",
        examples=["gemma", "deepseek"],
    )



class CitationItem(BaseModel):
    """Authoritative AI/ML knowledge citation with source attribution and formulas."""

    citation_code: str = Field(..., description="Short citation tag, e.g. [S1] or [Goodfellow-DL-§6.5]")
    source_title: str = Field(..., description="Book or paper title")
    section: str = Field(default="", description="Section or chapter reference")
    content_snippet: str = Field(..., description="Authoritative textbook/paper excerpt")
    formulas: list[str] = Field(default_factory=list, description="Canonical LaTeX formulas")
    file_name: Optional[str] = Field(default=None, description="PDF source filename")
    page_start: Optional[int] = Field(default=None, description="Physical PDF starting page")
    page_end: Optional[int] = Field(default=None, description="Physical PDF ending page")
    subject: Optional[str] = Field(default=None, description="Subject domain of source document")
    score: Optional[float] = Field(default=None, description="Cosine similarity score")
    source_url: Optional[str] = Field(default=None, description="Verified source URL if available")


class RoadmapResult(BaseModel):
    """Dynamic curriculum roadmap output with Mermaid.js diagram and downloadable Markdown."""

    domain: str = Field(..., description="Target curriculum domain (e.g. nlp_llms, computer_vision, all)")
    title: str = Field(..., description="Display title of roadmap")
    mermaid_code: str = Field(..., description="Syntactically valid Mermaid.js flowchart code")
    phases: list[str] = Field(default_factory=list, description="List of learning phase titles")
    total_nodes: int = Field(default=0, description="Total milestone nodes in roadmap")
    nodes: list[dict] = Field(default_factory=list, description="List of node milestone definitions")
    markdown_content: str = Field(..., description="Complete downloadable Markdown study guide")


class MasteryStatusUpdate(BaseModel):
    """Payload to update a student's learning state on a curriculum node."""

    session_id: str = Field(..., description="Session identifier")
    node_id: str = Field(..., description="Curriculum node ID, e.g. 'backprop_mlp'")
    status: str = Field(..., description="New status ('not_started', 'in_progress', 'mastered')")
    quiz_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional quiz evaluation score")


class EntityItem(BaseModel):
    """Schema for an extracted named entity."""

    text: str = Field(..., description="Extracted entity surface form text")
    label: str = Field(..., description="Entity category or tag (e.g., TECH_TERM, DATE)")
    start: Optional[int] = Field(default=None, description="Start character offset")
    end: Optional[int] = Field(default=None, description="End character offset")


class FuzzyQueryProfile(BaseModel):
    """Fuzzy Query Profile produced by Mamdani FIS for Gemma prompt orchestration."""

    technical_depth: float = Field(default=0.5, ge=0.0, le=1.0, description="Technical depth score")
    beginner_friendliness: float = Field(default=0.5, ge=0.0, le=1.0, description="Beginner friendliness score")
    conversationalness: float = Field(default=0.5, ge=0.0, le=1.0, description="Conversational style score")
    mathematical_detail: float = Field(default=0.3, ge=0.0, le=1.0, description="Mathematical detail score")
    example_preference: float = Field(default=0.6, ge=0.0, le=1.0, description="Example/analogy preference score")
    code_focus: float = Field(default=0.1, ge=0.0, le=1.0, description="Code focus score")
    aiml_relevance: float = Field(default=0.8, ge=0.0, le=1.0, description="AIML domain relevance score")
    explanation_depth: float = Field(default=0.5, ge=0.0, le=1.0, description="Explanation depth score")
    query_complexity: float = Field(default=0.5, ge=0.0, le=1.0, description="Query complexity score")
    ambiguity_level: float = Field(default=0.2, ge=0.0, le=1.0, description="Ambiguity level score")


class ChatResponse(BaseModel):
    """Comprehensive schema for chatbot response and NLP/Fuzzy routing metadata."""

    response: str = Field(..., description="Generated response text")
    layer: str = Field(default="gemma", description="Model engine ('gemma', 'layer2')")
    intent: str = Field(..., description="Classified intent tag")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence score")
    entities: list[EntityItem] = Field(default_factory=list, description="Extracted entities")
    similarity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Semantic similarity score")
    matched_pattern: Optional[str] = Field(default=None, description="Matched intent pattern")
    complexity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Query complexity score")
    entity_density: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Entity density score")
    activation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Mamdani fuzzy activation score")
    fuzzy_profile: Optional[FuzzyQueryProfile] = Field(default=None, description="Fuzzy Query Profile for prompt orchestration")
    routing_reason: Optional[str] = Field(default=None, description="Fuzzy router explanation")
    fallback: bool = Field(default=False, description="Whether fallback logic triggered")
    processing_time_ms: Optional[float] = Field(default=None, description="End-to-end processing latency in ms")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    model_name: Optional[str] = Field(default="Gemma 4 E2B Instruct", description="Underlying model name")
    retried: Optional[bool] = Field(default=False, description="Whether a generation retry occurred")
    tokens_generated: Optional[int] = Field(default=0, description="Tokens generated during inference")
    token_budget: Optional[int] = Field(default=None, description="Dynamically computed token generation limit")
    response_type: Optional[str] = Field(default=None, description="NLP-classified expected response type")
    demand_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Information volume demand score (0.0 to 1.0)")
    citations: list[CitationItem] = Field(default_factory=list, description="Authoritative RAG literature citations")
    roadmap: Optional[RoadmapResult] = Field(default=None, description="Dynamically generated curriculum roadmap if requested")
    socratic_mode: bool = Field(default=False, description="Whether response was generated under Socratic tutor guidance")
    depth_mode: Optional[str] = Field(default="auto", description="Pedagogical depth mode evaluated: auto, easy, medium, hard, ultra")


class QuizStartRequest(BaseModel):
    """Request payload to initiate a new quiz."""

    topic: str = Field(default="all", description="Target AIML topic or 'all'", examples=["NLP", "Deep Learning"])
    difficulty: str = Field(default="medium", description="Target difficulty ('easy', 'medium', 'hard', 'all')")
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of quiz questions requested")


class PublicQuizQuestion(BaseModel):
    """Public representation of a quiz question (correct_answer hidden)."""

    id: str = Field(..., description="Question unique ID")
    question: str = Field(..., description="Question prompt text")
    options: list[str] = Field(..., description="List of multiple-choice option strings")
    topic: str = Field(..., description="Question domain topic")
    difficulty: str = Field(..., description="Question difficulty level")
    current_index: int = Field(..., description="Zero-based question index in session")
    total_questions: int = Field(..., description="Total question count in quiz session")


class QuizStartResponse(BaseModel):
    """Response returned upon quiz creation."""

    session_id: str = Field(..., description="Unique session ID assigned to quiz")
    topic: str = Field(..., description="Selected quiz topic")
    difficulty: str = Field(..., description="Selected difficulty")
    total_questions: int = Field(..., description="Total number of questions in session")
    first_question: Optional[PublicQuizQuestion] = Field(default=None, description="First question payload")


class AnswerSubmitRequest(BaseModel):
    """Request payload when submitting a question answer."""

    answer_index: int = Field(..., ge=0, description="Zero-based index of chosen option")


class AnswerSubmitResponse(BaseModel):
    """Evaluation result returned after submitting an answer."""

    session_id: str = Field(..., description="Quiz session ID")
    question_id: str = Field(..., description="Answered question ID")
    user_answer: int = Field(..., description="Submitted answer index")
    correct_answer: int = Field(..., description="Correct option index")
    is_correct: bool = Field(..., description="Whether submitted answer was correct")
    explanation: str = Field(..., description="Educational explanation text")
    current_question_index: int = Field(..., description="Current zero-based index")
    total_questions: int = Field(..., description="Total question count")


class QuizScoreResponse(BaseModel):
    """Summary score breakdown for a completed or ongoing quiz session."""

    session_id: str = Field(..., description="Quiz session ID")
    topic: str = Field(..., description="Quiz topic")
    difficulty: str = Field(..., description="Quiz difficulty level")
    total_questions: int = Field(..., description="Total questions")
    correct_count: int = Field(..., description="Number of correctly answered questions")
    score_percent: float = Field(..., description="Score percentage (0.0 - 100.0)")
    completed: bool = Field(..., description="Whether quiz has been finished")


class HealthResponse(BaseModel):
    """Schema for health check endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(default="healthy", description="Current health status")
    model_loaded: bool = Field(..., description="Whether intent classifier model is loaded")
    intents_count: int = Field(..., description="Number of configured intents")
    layer2_enabled: bool = Field(default=True, description="Whether Layer 2 / generative model is active")
    active_sessions_count: int = Field(default=0, description="Active conversation sessions count")
    model_name: str = Field(default="Gemma 4 E2B Instruct", description="Active conversational model")
    gemma_loaded: bool = Field(default=False, description="Whether Gemma 4 model is loaded in memory")
    vram_allocated_mb: float = Field(default=0.0, description="Current allocated GPU VRAM in MB")


class RootResponse(BaseModel):
    """Schema for root endpoint."""

    message: str = Field(..., description="Status message")
    version: str = Field(..., description="Application version")
    docs_url: str = Field(default="/docs", description="Interactive OpenAPI documentation URL")
