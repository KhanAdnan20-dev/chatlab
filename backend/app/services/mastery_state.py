"""
Concept Mastery State Machine & Session Learning Tracker.

Tracks the student's mastery level across curriculum nodes,
recommends optimal next learning milestones, and records checkpoint quiz scores.
"""
import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from backend.app.services.roadmap_engine import CurriculumNode, roadmap_engine

logger = logging.getLogger(__name__)

STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_MASTERED = "mastered"


@dataclass
class NodeMasteryRecord:
    node_id: str
    status: str = STATUS_NOT_STARTED
    times_discussed: int = 0
    quiz_attempts: int = 0
    last_quiz_score: float = 0.0


@dataclass
class UserMasteryProfile:
    session_id: str
    nodes: Dict[str, NodeMasteryRecord] = field(default_factory=dict)

    def get_status(self, node_id: str) -> str:
        record = self.nodes.get(node_id)
        return record.status if record else STATUS_NOT_STARTED


class MasteryTracker:
    """
    Session-based student mastery manager across the AI/ML curriculum DAG.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, UserMasteryProfile] = {}

    def get_profile(self, session_id: str) -> UserMasteryProfile:
        """Retrieves or creates a student mastery profile."""
        if session_id not in self._profiles:
            profile = UserMasteryProfile(session_id=session_id)
            # Initialize with default graph nodes
            for nid in roadmap_engine.nodes_map.keys():
                profile.nodes[nid] = NodeMasteryRecord(node_id=nid, status=STATUS_NOT_STARTED)
            self._profiles[session_id] = profile
        return self._profiles[session_id]

    def update_node(
        self,
        session_id: str,
        node_id: str,
        status: str,
        quiz_score: Optional[float] = None,
    ) -> NodeMasteryRecord:
        """Updates a node's learning status and quiz metrics."""
        profile = self.get_profile(session_id)
        if node_id not in profile.nodes:
            profile.nodes[node_id] = NodeMasteryRecord(node_id=node_id)

        rec = profile.nodes[node_id]
        rec.status = status
        rec.times_discussed += 1

        if quiz_score is not None:
            rec.quiz_attempts += 1
            rec.last_quiz_score = quiz_score
            if quiz_score >= 0.70:
                rec.status = STATUS_MASTERED

        logger.info(
            "[MasteryTracker] session=%s node=%s status=%s score=%s",
            session_id, node_id, rec.status, quiz_score,
        )
        return rec

    def get_recommended_next(
        self,
        session_id: str,
        domain: str = "all",
    ) -> Optional[CurriculumNode]:
        """
        Finds the next optimal milestone: a node that is NOT mastered,
        but whose prerequisites ARE already mastered (or has no prerequisites).
        """
        profile = self.get_profile(session_id)
        nodes, _ = roadmap_engine.get_induced_subgraph(domain)

        # 1. Primary priority: nodes whose prerequisites are fulfilled (continuous mastery progression)
        for node in nodes:
            curr_status = profile.get_status(node.id)
            if curr_status != STATUS_MASTERED and node.prerequisites:
                if all(profile.get_status(pid) == STATUS_MASTERED for pid in node.prerequisites):
                    return node

        # 2. Secondary priority: unstarted root nodes (no prerequisites)
        for node in nodes:
            curr_status = profile.get_status(node.id)
            if curr_status != STATUS_MASTERED and not node.prerequisites:
                return node

        # If everything is mastered, return the last advanced node
        return nodes[-1] if nodes else None


# Singleton instance
mastery_tracker = MasteryTracker()
