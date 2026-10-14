"""FastAPI Router for Curriculum Roadmaps & Student Mastery State."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query, status
from pydantic import BaseModel

from backend.app.schemas import MasteryStatusUpdate, RoadmapResult
from backend.app.services.mastery_state import mastery_tracker
from backend.app.services.roadmap_engine import roadmap_engine

router = APIRouter(tags=["Curriculum & Mastery"])


@router.get(
    "/roadmap/generate",
    response_model=RoadmapResult,
    summary="Generate Dynamic Curriculum Roadmap",
    description="Traverses Curriculum DAG, resolves prerequisite ancestry, and returns Mermaid flowchart and Markdown study guide.",
)
def generate_roadmap(
    domain: str = Query(default="all", description="Target domain alias (e.g. 'nlp', 'vision', 'dl', 'all')"),
    query: str = Query(default="", description="Optional user inquiry context"),
) -> RoadmapResult:
    effective_query = query if query else domain
    rd = roadmap_engine.generate_roadmap(effective_query)
    return RoadmapResult(
        domain=rd.domain,
        title=rd.title,
        mermaid_code=rd.mermaid_code,
        phases=rd.phases,
        total_nodes=rd.node_count,
        markdown_content=rd.markdown_content,
        nodes=[
            {
                "id": n.id,
                "label": n.label,
                "domain": n.domain,
                "phase": n.phase,
                "difficulty": n.difficulty,
                "description": n.description,
                "key_topics": n.key_topics,
                "project": n.project,
                "paper": n.paper,
            }
            for n in rd.nodes
        ],
    )


class AIRoadmapRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    deep_synthesis: Optional[bool] = False


class AIRoadmapResponse(BaseModel):
    title: str
    domain: str
    markdown_content: str
    mermaid_code: str
    phases: List[str]
    total_nodes: int
    processing_time_ms: float
    nodes: List[Dict[str, Any]] = []


@router.post(
    "/roadmap/ai-generate",
    response_model=AIRoadmapResponse,
    summary="AI-Driven Dynamic Curriculum & Mermaid Flowchart Generator",
    description="Forces the LLM to generate a customized, rigorous syllabus and valid Mermaid.js flowchart for any custom topic.",
)
def ai_generate_roadmap(payload: AIRoadmapRequest) -> AIRoadmapResponse:
    import re
    import time
    from backend.app.services.chat_engine import EngineRequest, gemma_engine
    from backend.app.services.model_manager import model_manager

    t0 = time.perf_counter()
    query = payload.query.strip()
    if not query:
        query = "Artificial Intelligence & Soft Computing"

    base_rd = roadmap_engine.generate_roadmap(query)
    is_test = getattr(model_manager, "_test_mode", False)

    mermaid_code = base_rd.mermaid_code
    markdown_content = base_rd.markdown_content
    phases = base_rd.phases
    title = base_rd.title

    if getattr(payload, "deep_synthesis", False) and not is_test and (model_manager.gemma_available or model_manager.ensure_gemma_loaded()):
        try:
            req = EngineRequest(
                message=f"Generate a comprehensive, rigorous learning roadmap and syllabus for: {query}",
                intent="learning_path",
                confidence=1.0,
                session_id=payload.session_id or "",
            )
            resp = gemma_engine.generate(req)
            ai_text = resp.text

            m_match = re.search(r"```(?:mermaid)?\s*([\s\S]*?)```", ai_text)
            if m_match:
                raw_mermaid = m_match.group(1).strip()
                mermaid_code = roadmap_engine.sanitize_mermaid_code(raw_mermaid)
                clean_md = re.sub(r"```(?:mermaid)?\s*[\s\S]*?```", "", ai_text).strip()
                if clean_md:
                    markdown_content = clean_md
                    extracted_phases = re.findall(r"###?\s*(Phase\s*\d*[:\s][^\n]+)", clean_md, re.IGNORECASE)
                    if not extracted_phases:
                        extracted_phases = re.findall(r'subgraph\s+[\w\-]+\["([^"]+)"\]', mermaid_code)
                    if extracted_phases:
                        phases = [p.strip() for p in extracted_phases]
        except Exception as exc:
            pass

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    node_matches = re.findall(r'[\w\-]+\["[^"]+"\]', mermaid_code)
    node_count = len(node_matches) if node_matches else base_rd.node_count

    nodes_data = [
        {
            "id": n.id,
            "label": n.label,
            "domain": n.domain,
            "phase": n.phase,
            "difficulty": n.difficulty,
            "description": n.description,
            "key_topics": n.key_topics,
            "project": n.project,
            "paper": n.paper,
        }
        for n in base_rd.nodes
    ]

    return AIRoadmapResponse(
        title=f"{query.title()} Roadmap",
        domain=base_rd.domain,
        markdown_content=markdown_content,
        mermaid_code=mermaid_code,
        phases=phases if phases else ["Foundations", "Core Concepts", "Advanced Mastery"],
        total_nodes=node_count,
        processing_time_ms=elapsed_ms,
        nodes=nodes_data,
    )


@router.get(
    "/roadmap/curriculum",
    summary="Get Master Curriculum DAG",
    description="Returns all nodes, prerequisite edges, and domain categories in the curriculum graph.",
)
def get_curriculum() -> Dict[str, Any]:
    nodes = [
        {
            "id": n.id,
            "label": n.label,
            "domain": n.domain,
            "phase": n.phase,
            "prerequisites": n.prerequisites,
            "difficulty": n.difficulty,
            "description": n.description,
            "key_topics": n.key_topics,
            "project": n.project,
            "paper": n.paper,
        }
        for n in roadmap_engine.nodes_map.values()
    ]
    return {
        "nodes": nodes,
        "edges": roadmap_engine.edges,
        "total_nodes": len(nodes),
        "total_edges": len(roadmap_engine.edges),
    }


@router.get(
    "/mastery/{session_id}",
    summary="Get Student Mastery Profile",
    description="Returns student's concept mastery states across the curriculum for a given session.",
)
def get_mastery(session_id: str) -> Dict[str, Any]:
    profile = mastery_tracker.get_profile(session_id)
    recommended = mastery_tracker.get_recommended_next(session_id)
    return {
        "session_id": session_id,
        "nodes": {
            nid: {
                "status": rec.status,
                "times_discussed": rec.times_discussed,
                "quiz_attempts": rec.quiz_attempts,
                "last_quiz_score": rec.last_quiz_score,
            }
            for nid, rec in profile.nodes.items()
        },
        "recommended_next_node": {
            "id": recommended.id,
            "label": recommended.label,
            "domain": recommended.domain,
            "difficulty": recommended.difficulty,
            "description": recommended.description,
        } if recommended else None,
    }


@router.post(
    "/mastery/update",
    status_code=status.HTTP_200_OK,
    summary="Update Concept Mastery",
    description="Updates mastery state and quiz score for a curriculum node.",
)
def update_mastery(payload: MasteryStatusUpdate) -> Dict[str, Any]:
    rec = mastery_tracker.update_node(
        session_id=payload.session_id,
        node_id=payload.node_id,
        status=payload.status,
        quiz_score=payload.quiz_score,
    )
    return {
        "status": "success",
        "node_id": rec.node_id,
        "new_status": rec.status,
        "quiz_attempts": rec.quiz_attempts,
        "last_quiz_score": rec.last_quiz_score,
    }
