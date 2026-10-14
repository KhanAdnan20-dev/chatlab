"""Chat Orchestrator Service — Gemma 4 Conversational Engine Orchestrator with Mamdani Fuzzy Prompt Orchestration."""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.app.config import settings
from backend.app.schemas import CitationItem, EntityItem, FuzzyQueryProfile, RoadmapResult
from backend.app.services.chat_engine import EngineRequest, deepseek_engine, gemma_engine, vllm_engine
from backend.app.services.code_detector import code_detector


from backend.app.services.context_resolver import context_resolver
from backend.app.services.conversation_memory import conversation_memory
from backend.app.services.entity_extractor import entity_extractor
from backend.app.services.fuzzy_router import fuzzy_router
from backend.app.services.intent_classifier import intent_classifier
from backend.app.services.mastery_state import STATUS_IN_PROGRESS, mastery_tracker
from backend.app.services.model_manager import model_manager
from backend.app.services.preprocessing import preprocess
from backend.app.services.rag_engine import rag_engine
from backend.app.services.rag_retrieval_service import rag_retrieval_service
from backend.app.services.response_postprocessor import postprocess
from backend.app.services.roadmap_engine import roadmap_engine
from backend.app.services.semantic_similarity import semantic_similarity_engine
from backend.app.services.socratic_tutor import socratic_tutor

logger = logging.getLogger(__name__)


@dataclass
class ChatbotResult:
    """Structured result output produced by the Chat Orchestrator."""

    response: str
    layer: str
    intent: str
    confidence: float
    entities: list[EntityItem]
    similarity_score: float
    matched_pattern: str
    complexity_score: float
    entity_density: float
    activation_score: float
    routing_reason: str
    fallback: bool
    processing_time_ms: float
    session_id: str = ""
    fuzzy_profile: Optional[FuzzyQueryProfile] = None
    fired_rules: list[str] = field(default_factory=list)
    model_name: str = "Gemma 4 E2B Instruct"
    retried: bool = False
    tokens_generated: int = 0
    token_budget: Optional[int] = None
    response_type: Optional[str] = None
    demand_score: Optional[float] = None
    citations: list = field(default_factory=list)
    roadmap: Optional[RoadmapResult] = None
    socratic_mode: bool = False
    depth_mode: str = "auto"


class ChatbotService:
    """Central Chat Orchestrator powering Gemma 4 with Mamdani Fuzzy Prompt Tuning."""

    def __init__(self) -> None:
        self.confidence_threshold = settings.LAYER1_CONFIDENCE_THRESHOLD
        self.similarity_threshold = settings.SEMANTIC_SIMILARITY_THRESHOLD

    def process(
        self,
        message: str,
        session_id: Optional[str] = None,
        socratic_mode: bool = False,
        request_id: Optional[str] = None,
        depth_mode: str = "auto",
        model_choice: str = "gemma",
    ) -> ChatbotResult:
        """Executes full ChatLab NLP + Context + Mamdani Fuzzy Prompt Orchestration + Gemma Generation pipeline."""
        start_time = time.perf_counter()

        # Get or create conversation session
        session = conversation_memory.get_or_create(session_id)
        effective_session_id = session.session_id

        # Record user message in memory & fetch history
        conversation_memory.add_user_message(effective_session_id, message, request_id=request_id)
        history = conversation_memory.get_history(effective_session_id)

        has_context = False

        # 0A. PRIORITY CODE DETECTION FIRST
        code_res = code_detector.detect(message)
        if code_res.is_code and code_res.needs_clarification:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            conversation_memory.add_assistant_message(effective_session_id, code_res.clarification_message)
            return ChatbotResult(
                response=code_res.clarification_message,
                layer="gemma",
                intent="code_clarification",
                confidence=1.0,
                entities=[],
                similarity_score=0.0,
                matched_pattern="",
                complexity_score=0.20,
                entity_density=0.0,
                activation_score=0.20,
                routing_reason="Code snippet submitted without explicit user instructions.",
                fallback=False,
                processing_time_ms=elapsed_ms,
                session_id=effective_session_id,
                depth_mode=depth_mode,
            )

        effective_message = message

        # 0B. Context resolution for follow-ups
        if not code_res.is_code:
            preceding_history = history[:-1] if len(history) > 1 else []
            resolved_query = context_resolver.resolve(message, preceding_history)
            has_context = resolved_query.has_context

            if resolved_query.requires_clarification:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                conversation_memory.add_assistant_message(effective_session_id, resolved_query.clarification_message)
                return ChatbotResult(
                    response=resolved_query.clarification_message,
                    layer="gemma",
                    intent="clarification_needed",
                    confidence=1.0,
                    entities=[],
                    similarity_score=0.0,
                    matched_pattern="",
                    complexity_score=0.10,
                    entity_density=0.0,
                    activation_score=0.10,
                    routing_reason="Ambiguous follow-up query requiring user clarification.",
                    fallback=False,
                    processing_time_ms=elapsed_ms,
                    session_id=effective_session_id,
                    depth_mode=depth_mode,
                )

            effective_message = resolved_query.resolved_message

        # 1. Preprocess
        preprocessed = preprocess(effective_message)

        # 2. Intent classification (Layer 1 NLP Metadata)
        prediction = intent_classifier.predict(effective_message)
        primary_intent = prediction.intent
        confidence = prediction.confidence

        # 3. Entity extraction
        entities = entity_extractor.extract_entities(effective_message)

        # 4. Semantic similarity
        similarity = semantic_similarity_engine.find_best_match(effective_message)

        # 5. Intent Reconciliation & Discourse Connectives
        is_continuation = bool(re.match(
            r"^\s*(continue|go\s+on|keep\s+going|more|tell\s+me\s+more|proceed|next|carry\s+on|expand\s+on\s+that)\b",
            effective_message,
            re.IGNORECASE,
        ))
        is_surrender = bool(re.match(
            r"^\s*(idk|i\s+don'?t\s+know|no\s+idea|not\s+sure|unsure|i\s+have\s+no\s+idea|i\s+give\s+up|clueless|dont\s+know)\s*[\.!\?]?\s*$",
            effective_message,
            re.IGNORECASE,
        ))
        is_direct_req = bool(re.search(
            r"\b(just\s+explain(\s+it)?|explain\s+it(\s+directly)?|tell\s+me(\s+directly)?|"
            r"give\s+me\s+the\s+answer|direct\s+answer|stop\s+asking|just\s+tell\s+me|"
            r"no\s+more\s+questions|i\s+want\s+the\s+answer|answer\s+directly)\b",
            effective_message,
            re.IGNORECASE,
        ))

        resolved_intent = primary_intent
        if is_continuation:
            # Continuations must NEVER be misclassified as 'goodbye' or 'greeting'
            resolved_intent = "concept_explanation"
            confidence = 0.90
        elif is_direct_req:
            resolved_intent = "concept_explanation"
            confidence = 0.95
        elif is_surrender:
            resolved_intent = "concept_explanation"
            confidence = 0.85
        elif code_res.is_code:
            resolved_intent = code_res.suggested_intent or "code_explanation"
            confidence = max(confidence, 0.95)
        elif confidence < self.confidence_threshold:
            if similarity.score >= self.similarity_threshold and similarity.matched_intent and similarity.matched_intent != "unknown":
                resolved_intent = similarity.matched_intent
                confidence = max(confidence, similarity.score)

        # 6. NLP Feature extraction for Mamdani Fuzzy System
        complexity = fuzzy_router.compute_query_complexity(effective_message, preprocessed)
        if code_res.is_code:
            complexity = max(complexity, 0.90)

        entity_density = fuzzy_router.compute_entity_density(entities, preprocessed)
        conversationalness = fuzzy_router.compute_conversationalness(effective_message, resolved_intent, confidence)
        if code_res.is_code or has_context:
            conversationalness = 1.0

        # 7. Mamdani Fuzzy Inference System -> Fuzzy Query Profile
        fuzzy_decision = fuzzy_router.route(
            confidence=confidence,
            similarity=similarity.score,
            complexity=complexity,
            entity_density=entity_density,
            conversationalness=conversationalness,
            intent=resolved_intent,
            text=effective_message,
            is_code=code_res.is_code,
            has_context=has_context,
            depth_mode=depth_mode,
        )

        # 7.5. Dynamic Roadmap Extraction & Grounded RAG Retrieval
        is_roadmap = (
            "roadmap" in effective_message.lower()
            or "curriculum" in effective_message.lower()
            or "learning path" in effective_message.lower()
            or resolved_intent == "learning_path"
        )
        roadmap_res = None
        if is_roadmap:
            rd = roadmap_engine.generate_roadmap(effective_message)
            roadmap_res = RoadmapResult(
                domain=rd.domain,
                title=rd.title,
                mermaid_code=rd.mermaid_code,
                phases=rd.phases,
                total_nodes=rd.node_count,
                markdown_content=rd.markdown_content,
            )

        # RAG Retrieval Gatekeeper & Execution
        should_retrieve, retrieve_reason = rag_retrieval_service.should_retrieve(
            message=effective_message,
            intent=resolved_intent,
            confidence=confidence,
            fuzzy_profile=fuzzy_decision.fuzzy_profile,
            entities=entities,
            is_code=code_res.is_code,
        )

        citations = []
        rag_grounding = None
        if should_retrieve:
            citations, rag_grounding = rag_retrieval_service.retrieve(
                query=effective_message,
                top_k=settings.RAG_TOP_K,
            )
            logger.info(
                "[Orchestrator] RAG retrieval executed (%d passages). Reason: %s",
                len(citations),
                retrieve_reason,
            )
        else:
            logger.info("[Orchestrator] RAG retrieval bypassed. Reason: %s", retrieve_reason)

        # Socratic Scaffolding Directive
        socratic_directive = None
        if socratic_mode:
            soc_ctx = socratic_tutor.build_socratic_directive(effective_message, history=history)
            socratic_directive = soc_ctx.prompt_directive
            if soc_ctx.target_node:
                mastery_tracker.update_node(effective_session_id, soc_ctx.target_node.id, STATUS_IN_PROGRESS)

        # 8. Dispatch to Gemma 4 E2B Instruct Primary Conversational Engine
        engine_req = EngineRequest(
            message=effective_message,
            intent=resolved_intent,
            confidence=confidence,
            session_id=effective_session_id,
            history=history,
            is_code=code_res.is_code,
            code_text=code_res.code_text,
            fuzzy_profile=fuzzy_decision.fuzzy_profile,
            entities=entities,
            rag_grounding=rag_grounding,
            socratic_directive=socratic_directive,
            depth_mode=depth_mode,
        )

        use_deepseek = (model_choice or "").lower() in ("deepseek", "r1", "reason", "reasoning")
        use_vllm = not use_deepseek and bool(getattr(settings, "VLLM_ENABLED", False)) and vllm_engine.is_available() and not getattr(model_manager, "_test_mode", False)
        active_engine = deepseek_engine if use_deepseek else (vllm_engine if use_vllm else gemma_engine)
        engine_res = active_engine.generate(engine_req)
        reply_text = engine_res.text
        chosen_layer = engine_res.layer

        # 9. Response Postprocessing
        final_reply = postprocess(reply_text)

        # Record assistant reply in memory
        conversation_memory.add_assistant_message(effective_session_id, final_reply)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "[Orchestrator] session=%s intent=%s conf=%.2f layer=%s latency=%.1fms tokens=%d budget=%d type=%s",
            effective_session_id,
            resolved_intent,
            confidence,
            chosen_layer,
            elapsed_ms,
            engine_res.tokens_generated,
            engine_res.token_budget,
            engine_res.response_type,
        )

        return ChatbotResult(
            response=final_reply,
            layer=chosen_layer,
            intent=resolved_intent,
            confidence=confidence,
            entities=entities,
            similarity_score=similarity.score,
            matched_pattern=similarity.matched_pattern,
            complexity_score=complexity,
            entity_density=entity_density,
            activation_score=fuzzy_decision.activation_score,
            routing_reason=fuzzy_decision.reason,
            fallback=fuzzy_decision.is_fallback or ("fallback" in chosen_layer),
            processing_time_ms=elapsed_ms,
            session_id=effective_session_id,
            fuzzy_profile=fuzzy_decision.fuzzy_profile,
            fired_rules=fuzzy_decision.fired_rules,
            model_name="DeepSeek-R1 Distill Qwen 1.5B" if active_engine == deepseek_engine else "Gemma 4 E2B Instruct",
            retried=engine_res.retried,
            tokens_generated=engine_res.tokens_generated,
            token_budget=engine_res.token_budget,
            response_type=engine_res.response_type,
            demand_score=engine_res.demand_score,
            citations=citations,
            roadmap=roadmap_res,
            socratic_mode=socratic_mode,
            depth_mode=depth_mode,
        )

    def process_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        socratic_mode: bool = False,
        request_id: Optional[str] = None,
        depth_mode: str = "auto",
        model_choice: str = "gemma",
    ):
        """
        Executes ChatLab NLP analysis & Mamdani prompt builder, then progressively streams Gemma tokens.
        Yields chunk dictionaries:
          {"type": "token", "content": str}
          {"type": "metadata", ...}
        """
        start_time = time.perf_counter()

        # Get or create conversation session
        session = conversation_memory.get_or_create(session_id)
        effective_session_id = session.session_id

        # Record user message in memory & fetch history
        conversation_memory.add_user_message(effective_session_id, message, request_id=request_id)
        history = conversation_memory.get_history(effective_session_id)

        # Priority code detection
        code_res = code_detector.detect(message)
        if code_res.is_code and code_res.needs_clarification:
            conversation_memory.add_assistant_message(effective_session_id, code_res.clarification_message)
            yield {"type": "token", "content": code_res.clarification_message}
            yield {
                "type": "metadata",
                "response": code_res.clarification_message,
                "layer": "gemma",
                "intent": "code_clarification",
                "confidence": 1.0,
                "entities": [],
                "similarity_score": 0.0,
                "matched_pattern": "",
                "complexity_score": 0.20,
                "entity_density": 0.0,
                "activation_score": 0.20,
                "routing_reason": "Code snippet submitted without explicit user instructions.",
                "fallback": False,
                "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "session_id": effective_session_id,
                "fired_rules": [],
                "model_name": "Gemma 4 E2B Instruct",
                "retried": False,
                "tokens_generated": len(code_res.clarification_message.split()),
                "depth_mode": depth_mode,
            }
            return

        effective_message = message
        has_context = False
        if not code_res.is_code:
            preceding_history = history[:-1] if len(history) > 1 else []
            resolved_query = context_resolver.resolve(message, preceding_history)
            has_context = resolved_query.has_context

            if resolved_query.requires_clarification:
                conversation_memory.add_assistant_message(effective_session_id, resolved_query.clarification_message)
                yield {"type": "token", "content": resolved_query.clarification_message}
                yield {
                    "type": "metadata",
                    "response": resolved_query.clarification_message,
                    "layer": "gemma",
                    "intent": "clarification_needed",
                    "confidence": 1.0,
                    "entities": [],
                    "similarity_score": 0.0,
                    "matched_pattern": "",
                    "complexity_score": 0.10,
                    "entity_density": 0.0,
                    "activation_score": 0.10,
                    "routing_reason": "Ambiguous follow-up query requiring user clarification.",
                    "fallback": False,
                    "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "session_id": effective_session_id,
                    "fired_rules": [],
                    "model_name": "Gemma 4 E2B Instruct",
                    "retried": False,
                    "tokens_generated": len(resolved_query.clarification_message.split()),
                    "depth_mode": depth_mode,
                }
                return

            effective_message = resolved_query.resolved_message

        # Preprocessing & NLP Pipeline
        preprocessed = preprocess(effective_message)
        intent_res = intent_classifier.predict(effective_message)
        primary_intent = intent_res.intent
        confidence = intent_res.confidence

        entities = entity_extractor.extract_entities(effective_message)
        similarity = semantic_similarity_engine.find_best_match(effective_message)

        # Intent Reconciliation & Discourse Connectives
        is_continuation_s = bool(re.match(
            r"^\s*(continue|go\s+on|keep\s+going|more|tell\s+me\s+more|proceed|next|carry\s+on|expand\s+on\s+that)\b",
            effective_message,
            re.IGNORECASE,
        ))
        is_surrender_s = bool(re.match(
            r"^\s*(idk|i\s+don'?t\s+know|no\s+idea|not\s+sure|unsure|i\s+have\s+no\s+idea|i\s+give\s+up|clueless|dont\s+know)\s*[\.!\?]?\s*$",
            effective_message,
            re.IGNORECASE,
        ))
        is_direct_req_s = bool(re.search(
            r"\b(just\s+explain(\s+it)?|explain\s+it(\s+directly)?|tell\s+me(\s+directly)?|"
            r"give\s+me\s+the\s+answer|direct\s+answer|stop\s+asking|just\s+tell\s+me|"
            r"no\s+more\s+questions|i\s+want\s+the\s+answer|answer\s+directly)\b",
            effective_message,
            re.IGNORECASE,
        ))

        resolved_intent = primary_intent
        if is_continuation_s:
            resolved_intent = "concept_explanation"
            confidence = 0.90
        elif is_direct_req_s:
            resolved_intent = "concept_explanation"
            confidence = 0.95
        elif is_surrender_s:
            resolved_intent = "concept_explanation"
            confidence = 0.85
        elif code_res.is_code:
            resolved_intent = code_res.suggested_intent or "code_explanation"
            confidence = max(confidence, 0.95)
        elif confidence < self.confidence_threshold:
            if similarity.score >= self.similarity_threshold and similarity.matched_intent and similarity.matched_intent != "unknown":
                resolved_intent = similarity.matched_intent
                confidence = max(confidence, similarity.score)

        complexity = fuzzy_router.compute_query_complexity(effective_message, preprocessed)
        if code_res.is_code:
            complexity = max(complexity, 0.90)

        entity_density = fuzzy_router.compute_entity_density(entities, preprocessed)
        conversationalness = fuzzy_router.compute_conversationalness(effective_message, resolved_intent, confidence)
        if code_res.is_code or has_context:
            conversationalness = 1.0

        fuzzy_decision = fuzzy_router.route(
            confidence=confidence,
            similarity=similarity.score,
            complexity=complexity,
            entity_density=entity_density,
            conversationalness=conversationalness,
            intent=resolved_intent,
            text=effective_message,
            is_code=code_res.is_code,
            has_context=has_context,
            depth_mode=depth_mode,
        )

        # 7.5. Dynamic Roadmap Extraction & Grounded RAG Retrieval
        is_roadmap = (
            "roadmap" in effective_message.lower()
            or "curriculum" in effective_message.lower()
            or "learning path" in effective_message.lower()
            or resolved_intent == "learning_path"
        )
        roadmap_res = None
        if is_roadmap:
            rd = roadmap_engine.generate_roadmap(effective_message)
            roadmap_res = RoadmapResult(
                domain=rd.domain,
                title=rd.title,
                mermaid_code=rd.mermaid_code,
                phases=rd.phases,
                total_nodes=rd.node_count,
                markdown_content=rd.markdown_content,
            )

        # RAG Retrieval Gatekeeper & Execution
        should_retrieve, retrieve_reason = rag_retrieval_service.should_retrieve(
            message=effective_message,
            intent=resolved_intent,
            confidence=confidence,
            fuzzy_profile=fuzzy_decision.fuzzy_profile,
            entities=entities,
            is_code=code_res.is_code,
        )

        citations = []
        rag_grounding = None
        if should_retrieve:
            citations, rag_grounding = rag_retrieval_service.retrieve(
                query=effective_message,
                top_k=settings.RAG_TOP_K,
            )
            logger.info(
                "[Orchestrator] Stream RAG retrieval executed (%d passages). Reason: %s",
                len(citations),
                retrieve_reason,
            )
        else:
            logger.info("[Orchestrator] Stream RAG retrieval bypassed. Reason: %s", retrieve_reason)

        # Socratic Scaffolding Directive
        socratic_directive = None
        if socratic_mode:
            soc_ctx = socratic_tutor.build_socratic_directive(effective_message, history=history)
            socratic_directive = soc_ctx.prompt_directive
            if soc_ctx.target_node:
                mastery_tracker.update_node(effective_session_id, soc_ctx.target_node.id, STATUS_IN_PROGRESS)

        engine_req = EngineRequest(
            message=effective_message,
            intent=resolved_intent,
            confidence=confidence,
            session_id=effective_session_id,
            history=history,
            is_code=code_res.is_code,
            code_text=code_res.code_text,
            fuzzy_profile=fuzzy_decision.fuzzy_profile,
            entities=entities,
            rag_grounding=rag_grounding,
            socratic_directive=socratic_directive,
            depth_mode=depth_mode,
        )

        final_response_text = ""
        total_tokens = 0
        retried = False
        token_budget = 0
        response_type = "standard_concept"
        demand_score = 0.5

        use_deepseek = (model_choice or "").lower() in ("deepseek", "r1", "reason", "reasoning")
        use_vllm = not use_deepseek and bool(getattr(settings, "VLLM_ENABLED", False)) and vllm_engine.is_available() and not getattr(model_manager, "_test_mode", False)
        active_engine = deepseek_engine if use_deepseek else (vllm_engine if use_vllm else gemma_engine)
        streamed_pieces: list[str] = []
        for chunk in active_engine.generate_stream(engine_req):
            if chunk["type"] == "token":
                token_str = chunk["content"]
                streamed_pieces.append(token_str)
                yield {"type": "token", "content": token_str}
            elif chunk["type"] == "think":
                yield {"type": "think", "content": chunk["content"]}
            elif chunk["type"] == "done":
                final_response_text = chunk.get("response") or "".join(streamed_pieces)
                total_tokens = chunk.get("tokens_generated", len(streamed_pieces))
                retried = chunk.get("retried", False)
                token_budget = chunk.get("token_budget", 0)
                response_type = chunk.get("response_type", "standard_concept")
                demand_score = chunk.get("demand_score", 0.5)

        if not final_response_text and streamed_pieces:
            final_response_text = "".join(streamed_pieces)

        final_response_text = postprocess(final_response_text)

        # Record assistant reply in memory
        conversation_memory.add_assistant_message(effective_session_id, final_response_text)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        yield {
            "type": "metadata",
            "response": final_response_text,
            "layer": "deepseek" if active_engine == deepseek_engine else ("vllm" if active_engine == vllm_engine else "gemma"),
            "intent": resolved_intent,
            "confidence": confidence,
            "entities": [e.model_dump() if hasattr(e, "model_dump") else e for e in entities],
            "similarity_score": similarity.score,
            "matched_pattern": similarity.matched_pattern,
            "complexity_score": complexity,
            "entity_density": entity_density,
            "activation_score": fuzzy_decision.activation_score,
            "routing_reason": fuzzy_decision.reason,
            "fallback": fuzzy_decision.is_fallback,
            "processing_time_ms": elapsed_ms,
            "session_id": effective_session_id,
            "fuzzy_profile": fuzzy_decision.fuzzy_profile.model_dump() if fuzzy_decision.fuzzy_profile else None,
            "fired_rules": fuzzy_decision.fired_rules,
            "model_name": "DeepSeek-R1 Distill Qwen 1.5B" if active_engine == deepseek_engine else "Gemma 4 E2B Instruct",
            "retried": retried,
            "tokens_generated": total_tokens,
            "token_budget": token_budget,
            "response_type": response_type,
            "demand_score": demand_score,
            "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
            "roadmap": roadmap_res.model_dump() if roadmap_res else None,
            "socratic_mode": socratic_mode,
            "depth_mode": depth_mode,
        }


chatbot_service = ChatbotService()
