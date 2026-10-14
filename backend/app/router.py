"""Root router organizing Chat and Quiz sub-routers."""
from fastapi import APIRouter
from backend.app.api.chat_router import router as chat_router
from backend.app.api.quiz_router import router as quiz_router
from backend.app.api.roadmap_router import router as roadmap_router
from backend.app.config import settings
from backend.app.schemas import HealthResponse, RootResponse
from backend.app.services.conversation_memory import conversation_memory
from backend.app.services.intent_classifier import intent_classifier
from backend.app.services.model_manager import model_manager
from backend.app.services.response_engine import response_engine

router = APIRouter()

router.include_router(chat_router)
router.include_router(quiz_router)
router.include_router(roadmap_router)


@router.get(
    "/",
    response_model=RootResponse,
    summary="Root Endpoint",
    description="Returns basic application status and OpenAPI docs URL.",
)
def root() -> RootResponse:
    return RootResponse(
        message="ChatLab backend is running",
        version=settings.VERSION,
        docs_url="/docs",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check Endpoint",
    description="Returns backend operational health and model readiness status.",
)
def health() -> HealthResponse:
    status = model_manager.get_status()
    return HealthResponse(
        status="healthy",
        model_loaded=intent_classifier.is_loaded,
        intents_count=len(response_engine.responses_by_intent),
        layer2_enabled=True,
        active_sessions_count=conversation_memory.session_count(),
        model_name="Gemma 4 E2B Instruct",
        gemma_loaded=status.get("gemma_loaded", False),
        vram_allocated_mb=status.get("vram_allocated_mb", 0.0),
    )
