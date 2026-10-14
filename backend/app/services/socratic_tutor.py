"""
Socratic Tutoring Orchestrator & Active Recall Scaffolding.

Generates Socratic tutoring directives, guides students to discover concepts step-by-step,
and coordinates with the Concept Mastery DAG to identify prerequisite knowledge gaps.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

from backend.app.services.roadmap_engine import CurriculumNode, roadmap_engine

logger = logging.getLogger(__name__)


@dataclass
class SocraticContext:
    target_node: Optional[CurriculumNode]
    prerequisite_nodes: List[CurriculumNode]
    prompt_directive: str


class SocraticTutor:
    """
    Coordinates active recall pedagogical dialogues.
    """

    def _match_node_from_text(self, text: str) -> Optional[CurriculumNode]:
        """Matches a string to curriculum nodes via exact ID, label, or scored keyword overlap."""
        msg_lower = text.lower()
        # Direct ID or complete label match
        for nid, node in roadmap_engine.nodes_map.items():
            if nid in msg_lower or node.label.lower() in msg_lower:
                return node

        best_node = None
        max_score = 0
        for nid, node in roadmap_engine.nodes_map.items():
            score = 0
            for w in node.label.split():
                clean_w = w.lower().strip("(),:.-'")
                if len(clean_w) >= 4 and clean_w in msg_lower:
                    score += 2
            for topic in node.key_topics:
                if topic.lower() in msg_lower:
                    score += 3
                for w in topic.split():
                    clean_w = w.lower().strip("(),:.-'")
                    if len(clean_w) >= 5 and clean_w in msg_lower:
                        score += 1
            if score > max_score:
                max_score = score
                best_node = node
        return best_node if max_score > 0 else None

    def resolve_target_node(self, message: str, history: Optional[List[dict]] = None) -> Optional[CurriculumNode]:
        """Matches user message (or prior conversation turns if message is brief) to a curriculum node."""
        node = self._match_node_from_text(message)
        if node:
            return node

        # If message didn't match directly (e.g. "idk", "continue", "just explain it"), look back in history
        if history:
            for turn in reversed(history):
                content = turn.get("content", "")
                if content:
                    hist_node = self._match_node_from_text(content)
                    if hist_node:
                        return hist_node
        return None

    def build_socratic_directive(self, message: str, history: Optional[List[dict]] = None) -> SocraticContext:
        """
        Builds the system prompt directive for Socratic tutoring mode with adaptive pedagogical state.
        Detects student surrender, direct instruction requests, and continuations to prevent rigid deadlocks.
        """
        import re

        msg_clean = message.strip()
        target_node = self.resolve_target_node(message, history=history)
        prereqs: List[CurriculumNode] = []

        if target_node:
            prereqs = [
                roadmap_engine.nodes_map[pid]
                for pid in target_node.prerequisites
                if pid in roadmap_engine.nodes_map
            ]

        topic_label = target_node.label if target_node else "the active topic"

        # 1. Direct explanation requested ("just explain it", "tell me", "give me the answer")
        is_direct_explanation = bool(re.search(
            r"\b(just\s+explain(\s+it)?|explain\s+it(\s+directly)?|tell\s+me(\s+directly)?|"
            r"give\s+me\s+the\s+answer|direct\s+answer|stop\s+asking|just\s+tell\s+me|"
            r"no\s+more\s+questions|i\s+want\s+the\s+answer|answer\s+directly)\b",
            msg_clean,
            re.IGNORECASE,
        ))

        # 2. Continuation requested ("continue", "go on", "keep going", "more")
        is_continuation = bool(re.match(
            r"^\s*(continue|go\s+on|keep\s+going|more|tell\s+me\s+more|proceed|carry\s+on|expand\s+on\s+that)\b",
            msg_clean,
            re.IGNORECASE,
        ))

        # 3. Student surrender / struggle ("idk", "i don't know", "no idea", "unsure")
        is_surrender = bool(re.match(
            r"^\s*(idk|i\s+don'?t\s+know|no\s+idea|not\s+sure|unsure|i\s+have\s+no\s+idea|i\s+give\s+up|clueless|dont\s+know)\s*[\.!\?]?\s*$",
            msg_clean,
            re.IGNORECASE,
        ))

        if is_direct_explanation:
            directive = (
                "\n\n[SOCRATIC TUTOR PEDAGOGICAL MODE - DIRECT INSTRUCTION OVERRIDE]\n"
                f"The student has explicitly requested a direct explanation regarding '{topic_label}'.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Honor their request immediately: Transition from questioning to comprehensive Direct Instruction.\n"
                "2. Provide a thorough, step-by-step conceptual explanation with formulas, intuition, and architectural details.\n"
                "3. Do NOT withhold information, avoid answering, or reply with another roadblock question.\n"
                "4. Structure your response with clean Markdown headings and bullet points.\n"
                "5. Conclude with a supportive question asking if they would like a worked example or practice quiz.\n"
                "[END SOCRATIC MODE]\n"
            )
        elif is_continuation:
            directive = (
                "\n\n[SOCRATIC TUTOR PEDAGOGICAL MODE - SEAMLESS CONTINUATION]\n"
                f"The student asked to continue the explanation on '{topic_label}'.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Seamlessly resume the ongoing explanation directly from the previous incomplete point.\n"
                "2. Do NOT repeat introductory phrases, conversational pleasantries, or greetings.\n"
                "3. Deliver the remaining formulas, architecture components, and theoretical insights completely.\n"
                "[END SOCRATIC MODE]\n"
            )
        elif is_surrender:
            directive = (
                "\n\n[SOCRATIC TUTOR PEDAGOGICAL MODE - GUIDED SCAFFOLDING & HINT]\n"
                f"The student is unsure or stuck on the current concept ('{topic_label}').\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Encourage them warmly and normalize that this is a challenging topic.\n"
                "2. Provide an intuitive, real-world analogy or a concrete hint that partially reveals the core concept.\n"
                "3. Ask a simpler, bite-sized multiple-choice or true/false question to help them take the first step.\n"
                "[END SOCRATIC MODE]\n"
            )
        elif target_node and prereqs:
            prereq_labels = ", ".join(f"'{p.label}'" for p in prereqs)
            directive = (
                "\n\n[SOCRATIC TUTOR PEDAGOGICAL MODE ACTIVATED]\n"
                f"The student is inquiring about '{target_node.label}'.\n"
                f"Its essential prerequisite knowledge foundations are: {prereq_labels}.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. DO NOT give away the complete final answer, full theorem, or code solution upfront.\n"
                "2. Instead, guide the student with Socratic scaffolding: ask 1 or 2 focused, intuitive questions "
                f"grounded in the prerequisite concept ({prereq_labels}) to help them deduce the key insight themselves.\n"
                "3. Keep your response focused and end with a clear question for the user.\n"
                "4. Be encouraging, warm, and pedagogically rigorous.\n"
                "[END SOCRATIC MODE]\n"
            )
        else:
            directive = (
                "\n\n[SOCRATIC TUTOR PEDAGOGICAL MODE ACTIVATED]\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. DO NOT lecture or output an exhaustive answer immediately.\n"
                "2. Lead the student through guided inquiry: ask a clarifying question about the core intuition first.\n"
                "3. End with an invitation for them to explain what they think happens.\n"
                "[END SOCRATIC MODE]\n"
            )

        return SocraticContext(
            target_node=target_node,
            prerequisite_nodes=prereqs,
            prompt_directive=directive,
        )


# Singleton instance
socratic_tutor = SocraticTutor()
