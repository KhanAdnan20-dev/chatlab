"""FastAPI Application Entry Point for ChatLab Monolith."""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.exceptions import ChatLabError, to_http
from backend.app.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ChatLab")

import os
from contextlib import asynccontextmanager

# Set HuggingFace offline mode to eliminate WAN roundtrips on runtime resolution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm lightweight NLP engines at boot to prevent first-query stalls
    try:
        from backend.app.services.semantic_similarity import semantic_similarity_engine
        from backend.app.services.intent_classifier import intent_classifier
        intent_classifier.predict("hello")
        semantic_similarity_engine.compute_similarity("what is machine learning")
        logger.info("NLP feature engines pre-warmed successfully.")
    except Exception as e:
        logger.warning("NLP pre-warming skipped: %s", e)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ChatLabError)
def chatlab_domain_exception_handler(request: Request, exc: ChatLabError):
    http_exc = to_http(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content={"detail": http_exc.detail},
    )

app.include_router(router)
