from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Schema for incoming user chat requests."""

    message: str = Field(
        ...,
        min_length=1,
        description="Natural-language input text message from the user",
        examples=["Tell me about your capabilities", "I have a Python exam on Monday."],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for multi-turn tracking",
        examples=["session-1234"],
    )


class EntityItem(BaseModel):
    """Schema for an extracted named entity."""

    text: str = Field(..., description="Extracted entity surface form text")
    label: str = Field(..., description="Entity category or tag (e.g., TECH_TERM, DATE, CONCEPT)")
    start: Optional[int] = Field(default=None, description="Start character offset in original text")
    end: Optional[int] = Field(default=None, description="End character offset in original text")


class ChatResponse(BaseModel):
    """Comprehensive schema for chatbot output, Layer 1 NLP metadata, and Fuzzy Routing diagnostics."""

    response: str = Field(..., description="Generated natural-language response message")
    layer: str = Field(
        ...,
        description="System layer that generated the response ('layer1', 'fallback', or 'layer2_pending')",
        examples=["layer1"],
    )
    intent: str = Field(
        ...,
        description="Classified intent tag",
        examples=["greeting", "programming_help"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification probability/confidence score for the chosen intent",
        examples=[0.92],
    )
    entities: list[EntityItem] = Field(
        default_factory=list,
        description="List of entities recognized in the input text",
    )
    similarity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score against closest pattern in intent dataset",
        examples=[0.88],
    )
    matched_pattern: Optional[str] = Field(
        default=None,
        description="Closest matching training pattern from dataset",
        examples=["how do i learn python"],
    )
    complexity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Computed query lexical complexity score in [0, 1]",
        examples=[0.18],
    )
    entity_density: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized ratio of recognized entities to tokens in [0, 1]",
        examples=[0.33],
    )
    activation_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Defuzzified Layer Activation Score from Mamdani FIS in [0, 1]",
        examples=[0.21],
    )
    routing_reason: Optional[str] = Field(
        default=None,
        description="Fuzzy inference rationale for layer selection",
        examples=["Fuzzy activation score (0.21 <= 0.50) confirms Layer 1 suitability."],
    )
    fallback: bool = Field(
        default=False,
        description="Whether fallback logic was triggered due to high fuzzy activation or low confidence",
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Total end-to-end processing latency in milliseconds",
        examples=[14.2],
    )


class HealthResponse(BaseModel):
    """Schema for health check endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(default="healthy", description="Current service health status")
    model_loaded: bool = Field(..., description="Indicates whether the intent model is loaded in memory")
    intents_count: int = Field(..., description="Total number of active intents configured")


class RootResponse(BaseModel):
    """Schema for root endpoint."""

    message: str = Field(..., description="Status message indicating backend is active")
    version: str = Field(..., description="Application version")
    docs_url: str = Field(default="/docs", description="Interactive API documentation URL")
