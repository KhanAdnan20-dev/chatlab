"""FastAPI Router for Chat endpoints."""
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas import ChatRequest, ChatResponse
from backend.app.services.chatbot import chatbot_service

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with ChatLab Orchestrator",
    description=(
        "Processes user message through NLP pipeline, Mamdani Fuzzy Prompt Tuning, and Google Gemma 4 E2B Instruct."
    ),
)
def chat(request: ChatRequest) -> ChatResponse:
    clean_msg = request.message.strip()
    if not clean_msg:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message field cannot be empty or whitespace.",
        )

    result = chatbot_service.process(
        message=clean_msg,
        session_id=request.session_id,
        socratic_mode=request.socratic_mode,
        request_id=request.request_id,
        depth_mode=request.depth_mode,
        model_choice=request.model_choice,
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
        session_id=result.session_id,
        fuzzy_profile=result.fuzzy_profile,
        model_name=result.model_name,
        retried=result.retried,
        tokens_generated=result.tokens_generated,
        token_budget=result.token_budget,
        response_type=result.response_type,
        demand_score=result.demand_score,
        citations=result.citations,
        roadmap=result.roadmap,
        socratic_mode=result.socratic_mode,
        depth_mode=result.depth_mode,
    )


@router.post(
    "/stream",
    summary="Stream Chat with Gemma 4 E2B Instruct",
    description="Progressively streams generated tokens using Server-Sent Events (SSE).",
)
def chat_stream(request: ChatRequest):
    import json
    from fastapi.responses import StreamingResponse

    clean_msg = request.message.strip()
    if not clean_msg:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message field cannot be empty or whitespace.",
        )

    from fastapi.encoders import jsonable_encoder

    def event_generator():
        for chunk in chatbot_service.process_stream(
            message=clean_msg,
            session_id=request.session_id,
            socratic_mode=request.socratic_mode,
            request_id=request.request_id,
            depth_mode=request.depth_mode,
            model_choice=request.model_choice,
        ):
            yield f"data: {json.dumps(jsonable_encoder(chunk))}\n\n"


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
