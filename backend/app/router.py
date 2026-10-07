from fastapi import APIRouter, HTTPException, status
from backend.app.config import settings
from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RootResponse,
)
from backend.app.services.chatbot import chatbot_service
from backend.app.services.intent_classifier import intent_classifier
from backend.app.services.response_engine import response_engine

router = APIRouter()


@router.get(
    "/",
    response_model=RootResponse,
    summary="Root Endpoint",
    description="Returns basic application status and metadata.",
)
def root() -> RootResponse:
    """Root endpoint to verify that the ChatLab backend is active."""
    return RootResponse(
        message="ChatLab backend is running",
        version=settings.VERSION,
        docs_url="/docs",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns backend health status, ML model loaded status, and number of active intents.",
)
def health() -> HealthResponse:
    """Health check endpoint to verify ML model readiness and operational health."""
    return HealthResponse(
        status="healthy",
        model_loaded=intent_classifier.is_loaded,
        intents_count=len(response_engine.responses_by_intent),
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with ChatLab",
    description=(
        "Processes natural-language queries through the Layer 1 NLP pipeline and Mamdani Fuzzy Router: "
        "preprocessing, intent classification, named entity recognition, semantic similarity, "
        "complexity analysis, fuzzy decision routing, and Layer 1 response generation."
    ),
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint.
    Accepts user message, executes NLP pipeline and Mamdani Fuzzy Logic router,
    and returns reply with comprehensive diagnostics.
    """
    clean_msg = request.message.strip()
    if not clean_msg:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message field cannot be empty or solely whitespace.",
        )

    result = chatbot_service.process(
        message=clean_msg,
        session_id=request.session_id,
    )

    return ChatResponse(
        response=result.response,
        layer=result.layer,
        intent=result.intent,
        confidence=result.confidence,
        entities=result.entities,
        similarity_score=result.similarity_score,
        matched_pattern=result.matched_pattern,
        complexity_score=result.complexity_score,
        entity_density=result.entity_density,
        activation_score=result.activation_score,
        routing_reason=result.routing_reason,
        fallback=result.fallback,
        processing_time_ms=result.processing_time_ms,
    )
