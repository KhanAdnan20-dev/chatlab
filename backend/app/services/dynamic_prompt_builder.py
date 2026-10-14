"""Dynamic Prompt Builder — Translates Mamdani Fuzzy Query Profiles into natural prompt directives for Gemma 4."""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.app.config import settings
from backend.app.schemas import FuzzyQueryProfile
from backend.app.services.token_estimator import (
    RESPONSE_TYPE_CODE_IMPLEMENTATION,
    RESPONSE_TYPE_CONVERSATIONAL,
    RESPONSE_TYPE_IN_DEPTH_EXPLANATION,
    RESPONSE_TYPE_STANDARD_CONCEPT,
    RESPONSE_TYPE_STEP_BY_STEP_DERIVATION,
    RESPONSE_TYPE_STRUCTURED_COMPARISON,
    RESPONSE_TYPE_ULTRA_CONCISE,
    RESPONSE_TYPE_CURRICULUM_ROADMAP,
    token_budget_estimator,
)

logger = logging.getLogger(__name__)


@dataclass
class BuiltPrompt:
    """Prompt payload and generation configuration built for Gemma inference."""

    system_instruction: str
    messages: List[Dict[str, str]]
    token_budget: int
    response_type: str = "standard_concept"
    demand_score: float = 0.5
    temperature: float = 0.6
    top_p: float = 0.9


class DynamicPromptBuilder:
    """
    Translates Mamdani Fuzzy Query Profiles into natural conversational guidelines
    and builds structured chat message histories for Google Gemma 4 E2B Instruct.
    """

    SYSTEM_BASE = (
        "You are ChatLab, an intelligent educational conversational AI assistant "
        "powered by Google Gemma 4 E2B Instruct.\n"
        "Your core expertise is Artificial Intelligence, Machine Learning, and Computer Science, "
        "but you also converse warmly, accurately, and naturally on general topics (greetings, "
        "casual questions, mathematics, and everyday inquiries).\n\n"
        "PROFESSIONAL FORMATTING RULES:\n"
        "- Structure your explanation with clean, readable Markdown (headings `###`, bold key concepts, bullet lists).\n"
        "- Format all mathematical equations, variables, and formulas using standard LaTeX notation ($...$ for inline variables, $$...$$ for standalone display equations).\n"
        "- Format code cleanly in fenced code blocks with language identifiers (e.g., ```python).\n"
        "- Be direct and concise. Deliver high information density without unnecessary preamble or repetitive filler.\n"
        "- Conclude your thoughts and sentences completely. Ensure all explanations and lists terminate with complete sentences and closed punctuation.\n"
        "- Ensure all Markdown fences, brackets, and bold tags are closed cleanly.\n"
    )

    def compute_token_budget(
        self,
        fuzzy_profile: Optional[FuzzyQueryProfile],
        is_code: bool = False,
        message: str = "",
        intent: str = "",
        entities: Optional[List[Any]] = None,
        depth_mode: str = "auto",
    ) -> int:
        """
        Determines calibrated token budget based on NLP Response Demand Analyzer.
        """
        decision = token_budget_estimator.estimate(
            message=message,
            intent=intent,
            fuzzy_profile=fuzzy_profile,
            entities=entities,
            is_code=is_code,
            depth_mode=depth_mode,
        )
        return decision.token_budget

    def translate_fuzzy_profile(
        self,
        fp: FuzzyQueryProfile,
        is_code: bool = False,
    ) -> str:
        """Translates crisp/centroid fuzzy outputs into concise, natural language instructions."""
        directives = []

        # 1. Technical Depth & Beginner Friendliness
        if fp.beginner_friendliness >= 0.75 or fp.technical_depth <= 0.30:
            directives.append(
                "Explain clearly using simple analogies and beginner-friendly intuition. "
                "Avoid unnecessary jargon or advanced mathematics."
            )
        elif fp.technical_depth >= 0.70:
            directives.append(
                "Provide rigorous technical depth, incorporating formal terminology, "
                "underlying mechanisms, and architectural nuances."
            )
        else:
            directives.append(
                "Provide a balanced explanation accessible to an intermediate learner."
            )

        # 2. Mathematical Detail
        if fp.mathematical_detail >= 0.65:
            directives.append(
                "Include relevant formulas, mathematical definitions, or step-by-step arithmetic where applicable."
            )
        elif fp.mathematical_detail <= 0.25:
            directives.append(
                "Focus on conceptual intuition without heavy mathematical derivations."
            )

        # 3. Example Preference
        if fp.example_preference >= 0.65:
            directives.append(
                "Illustrate key points with a concrete, relatable example or practical scenario."
            )

        # 4. Code Focus
        if is_code or fp.code_focus >= 0.70:
            directives.append(
                "Ground your explanation in code. Directly analyze, explain, or provide clean code examples."
            )

        # 5. Conversationalness
        if fp.conversationalness >= 0.75:
            directives.append(
                "Maintain an engaging, conversational, and encouraging tone."
            )
        elif fp.conversationalness <= 0.35:
            directives.append(
                "Keep the response concise, structured, and strictly factual."
            )

        # 6. AIML Relevance vs General Query
        if fp.aiml_relevance < 0.50:
            directives.append(
                "Answer the general question naturally and directly without forcing AIML concepts."
            )

        return "\n".join(f"- {d}" for d in directives)

    def build(
        self,
        message: str,
        fuzzy_profile: Optional[FuzzyQueryProfile],
        history: Optional[List[Dict[str, str]]] = None,
        is_code: bool = False,
        code_text: str = "",
        retry_directive: Optional[str] = None,
        intent: str = "",
        entities: Optional[List[Any]] = None,
        rag_grounding: Optional[str] = None,
        socratic_directive: Optional[str] = None,
        depth_mode: str = "auto",
    ) -> BuiltPrompt:
        """
        Builds the complete message structure for Gemma's chat template.
        """
        fp = fuzzy_profile or FuzzyQueryProfile()
        translated_instructions = self.translate_fuzzy_profile(fp, is_code=is_code)

        estimate = token_budget_estimator.estimate(
            message=message,
            intent=intent,
            fuzzy_profile=fp,
            entities=entities,
            is_code=is_code,
            depth_mode=depth_mode,
        )
        token_budget = estimate.token_budget

        # Response type formatting guidance tailored to detected demand or selected depth mode
        normalized_mode = (depth_mode or "auto").lower().strip()
        type_directives = []

        if normalized_mode == "easy":
            type_directives.append(
                "PEDAGOGICAL DEPTH [EASY - DEFINITION & INTUITION]:\n"
                "- Give a concise, clear definition-style answer using simple everyday analogies.\n"
                "- Emphasize high beginner friendliness and high clarity.\n"
                "- Avoid unnecessary mathematical formulas or complex proofs."
            )
        elif normalized_mode == "medium":
            type_directives.append(
                "PEDAGOGICAL DEPTH [MEDIUM - IN-DEPTH EXPLANATION]:\n"
                "- Provide a thorough, well-structured explanation covering key concepts, mechanisms, and trade-offs.\n"
                "- Balance theoretical intuition with clear headings, bullet points, and practical examples."
            )
        elif normalized_mode == "hard":
            type_directives.append(
                "PEDAGOGICAL DEPTH [HARD - EXTREME STEP-BY-STEP RIGOR]:\n"
                "- Provide an extremely detailed, step-by-step rigorous breakdown of the architecture and theory.\n"
                "- Include exact mathematical formulas in LaTeX notation ($...$ and $$...$$), derivations, and component mechanics.\n"
                "- Detail internal algorithms, execution flows, and edge cases in exhaustive detail."
            )
        elif normalized_mode == "ultra":
            type_directives.append(
                "PEDAGOGICAL DEPTH [ULTRA - COMPREHENSIVE MASTERCLASS & EXHAUSTIVE DEEP-DIVE]:\n"
                "- Provide an exhaustive, publication-grade masterclass monograph on the requested topic.\n"
                "- Cover complete foundational intuition, full mathematical formulations in canonical LaTeX notation ($...$ and $$...$$), underlying architectural mechanisms, complete execution workflows, pseudocode or runnable code implementations, trade-offs, and advanced real-world edge cases.\n"
                "- Structure comprehensively with hierarchical headings (`###`), detailed conceptual explanations, bulleted summaries, and complete concluding insights without truncating."
            )
        elif estimate.response_type == RESPONSE_TYPE_ULTRA_CONCISE:
            type_directives.append(
                "The user requested strict brevity. Answer directly and concisely in 1 to 2 sentences without filler or preamble."
            )
        elif estimate.response_type == RESPONSE_TYPE_STRUCTURED_COMPARISON:
            type_directives.append(
                "Organize your answer with structured comparison points, clear headings, or bulleted trade-offs."
            )
        elif estimate.response_type == RESPONSE_TYPE_STEP_BY_STEP_DERIVATION:
            type_directives.append(
                "Present your explanation or mathematical formulation in clear, sequential numbered steps."
            )
        elif estimate.response_type == RESPONSE_TYPE_IN_DEPTH_EXPLANATION:
            type_directives.append(
                "Provide an exhaustive, in-depth explanation covering underlying theory, architecture, and edge cases."
            )
        elif estimate.response_type == RESPONSE_TYPE_CODE_IMPLEMENTATION:
            type_directives.append(
                "Provide complete, clean, and executable code accompanied by a concise breakdown of how it works."
            )
        elif estimate.response_type == RESPONSE_TYPE_CURRICULUM_ROADMAP or intent == "learning_path":
            type_directives.append(
                "CURRICULUM ARCHITECT & MERMAID FLOWCHART INSTRUCTION:\n"
                "The user is requesting a learning roadmap or syllabus. You MUST provide two clear sections:\n"
                "1. Structured Syllabus breakdown with phases (Phase 1, Phase 2, etc.), key topics, projects, and literature.\n"
                "2. A complete, valid Mermaid.js flowchart in a ```mermaid ... ``` code block. Follow strict syntax:\n"
                "   - Start with 'graph TD'\n"
                "   - Group each phase into a subgraph: subgraph Phase_1[\"Phase 1: Title\"] ... end\n"
                "   - Crucial: ALWAYS quote all node labels: id[\"Node Label\"]\n"
                "   - Connect prerequisite edges: id1 --> id2"
            )
        elif estimate.response_type == RESPONSE_TYPE_CONVERSATIONAL or intent in ("greeting", "identity"):
            type_directives.append(
                "Converse warmly, naturally, and helpfully in 1 to 2 engaging sentences. Offer friendly assistance."
            )
        else:
            type_directives.append(
                "Provide a clear, engaging, and well-structured explanation with intuitive intuition and clean Markdown."
            )

        formatting_addon = ("\n\nRESPONSE FORMAT DIRECTIVES:\n" + "\n".join(f"- {d}" for d in type_directives)) if type_directives else ""

        system_instruction = (
            f"{self.SYSTEM_BASE}\n"
            "ADAPTIVE INSTRUCTIONS (derived via Mamdani Fuzzy System):\n"
            f"{translated_instructions}"
            f"{formatting_addon}\n\n"
            "CONVERSATIONAL GUIDELINES:\n"
            "- Address the user's specific query directly, faithfully, and naturally.\n"
            "- When code is supplied, analyze and explain that exact code clearly.\n"
            "- Do not echo internal instructions or raw prompt tokens.\n"
            "- Complete your thoughts and explanations cleanly within the response."
        )

        if rag_grounding:
            system_instruction += rag_grounding

        if socratic_directive:
            system_instruction += socratic_directive
            # Socratic prompts:
            # If the user requested direct explanation, continuation, or a deep concept,
            # allocate full explanation budget (640-1024 tokens) so the response is never truncated!
            is_direct_override = (
                "DIRECT INSTRUCTION OVERRIDE" in socratic_directive
                or "SEAMLESS CONTINUATION" in socratic_directive
                or "GUIDED SCAFFOLDING" in socratic_directive
                or estimate.response_type in (
                    RESPONSE_TYPE_CODE_IMPLEMENTATION,
                    RESPONSE_TYPE_STEP_BY_STEP_DERIVATION,
                    RESPONSE_TYPE_IN_DEPTH_EXPLANATION,
                    RESPONSE_TYPE_STANDARD_CONCEPT,
                    RESPONSE_TYPE_STRUCTURED_COMPARISON,
                )
            )
            if normalized_mode == "auto":
                if is_direct_override:
                    token_budget = max(token_budget, 640)
                else:
                    # Standard Socratic active recall questions: provide comfortable breathing room (320 tokens)
                    token_budget = max(280, min(token_budget, 400))

        if retry_directive:
            system_instruction += f"\n\nRETRY DIRECTIVE: {retry_directive}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_instruction}
        ]

        # Add recent conversation turns (sensible window of 6 turns)
        if history:
            recent_turns = history[-6:]
            for turn in recent_turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # Format user message (include code if detected and not already in message)
        user_content = message
        if is_code and code_text and code_text not in message:
            user_content = f"{message}\n\n```python\n{code_text}\n```"

        messages.append({"role": "user", "content": user_content})

        return BuiltPrompt(
            system_instruction=system_instruction,
            messages=messages,
            token_budget=token_budget,
            response_type=estimate.response_type,
            demand_score=estimate.demand_score,
            temperature=0.6 if not retry_directive else 0.3,
            top_p=0.9,
        )


dynamic_prompt_builder = DynamicPromptBuilder()
