"""
NLP-Driven Response Demand Analyzer & Dynamic Token Budget Estimator.

Analyzes linguistic cues, syntactic structure (POS tagging), entity density,
user-requested brevity/depth constraints, and Mamdani Fuzzy Query Profiles
to determine the precise response type and calibrate the optimal token generation budget.
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from backend.app.schemas import EntityItem, FuzzyQueryProfile

try:
    import nltk
    from nltk import pos_tag, word_tokenize
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


# Standard response categories
RESPONSE_TYPE_ULTRA_CONCISE = "ultra_concise"
RESPONSE_TYPE_CONVERSATIONAL = "conversational"
RESPONSE_TYPE_STANDARD_CONCEPT = "standard_concept"
RESPONSE_TYPE_STRUCTURED_COMPARISON = "structured_comparison"
RESPONSE_TYPE_STEP_BY_STEP_DERIVATION = "step_by_step_derivation"
RESPONSE_TYPE_CODE_IMPLEMENTATION = "code_implementation"
RESPONSE_TYPE_IN_DEPTH_EXPLANATION = "in_depth_explanation"
RESPONSE_TYPE_CURRICULUM_ROADMAP = "curriculum_roadmap"

# Baseline token budgets per category
CATEGORY_BASE_BUDGETS = {
    RESPONSE_TYPE_ULTRA_CONCISE: 72,
    RESPONSE_TYPE_CONVERSATIONAL: 64,
    RESPONSE_TYPE_STANDARD_CONCEPT: 512,
    RESPONSE_TYPE_STRUCTURED_COMPARISON: 640,
    RESPONSE_TYPE_STEP_BY_STEP_DERIVATION: 768,
    RESPONSE_TYPE_CODE_IMPLEMENTATION: 896,
    RESPONSE_TYPE_IN_DEPTH_EXPLANATION: 896,
    RESPONSE_TYPE_CURRICULUM_ROADMAP: 1024,
}

# Bounds per category (min_tokens, max_tokens)
CATEGORY_BOUNDS = {
    RESPONSE_TYPE_ULTRA_CONCISE: (48, 96),
    RESPONSE_TYPE_CONVERSATIONAL: (48, 80),
    RESPONSE_TYPE_STANDARD_CONCEPT: (256, 640),
    RESPONSE_TYPE_STRUCTURED_COMPARISON: (384, 768),
    RESPONSE_TYPE_STEP_BY_STEP_DERIVATION: (512, 896),
    RESPONSE_TYPE_CODE_IMPLEMENTATION: (512, 1024),
    RESPONSE_TYPE_IN_DEPTH_EXPLANATION: (512, 1024),
    RESPONSE_TYPE_CURRICULUM_ROADMAP: (768, 1280),
}


@dataclass
class TokenBudgetDecision:
    """Detailed result produced by the TokenBudgetEstimator."""

    token_budget: int
    response_type: str
    demand_score: float  # 0.0 (minimal brevity) to 1.0 (exhaustive deep dive)
    reasoning: str
    base_budget: int
    multiplier: float


class TokenBudgetEstimator:
    """
    NLP Response Demand Analyzer.
    Determines what type of response a question wants and computes an optimal,
    dynamic token budget so that short questions generate quickly and complex
    explanations have ample room without truncation.
    """

    # Lexical patterns for explicit formatting and length constraints
    RE_ULTRA_CONCISE = re.compile(
        r"\b(in\s+one\s+sentence|in\s+1\s+sentence|in\s+one\s+line|in\s+a\s+nutshell|"
        r"briefly|short\s+answer|tl;?dr|quick\s+definition|single\s+sentence|"
        r"in\s+\d+\s+words|concise\s+summary|yes\s+or\s+no|brief\s+summary)\b",
        re.IGNORECASE,
    )

    RE_IN_DEPTH = re.compile(
        r"\b(in[- ]depth|in[- ]detail|detailed|deep[- ]dive|comprehensive|"
        r"thoroughly|elaborate|everything\s+about|exhaustive|complete\s+guide|"
        r"rigorous|full\s+explanation|extensive)\b",
        re.IGNORECASE,
    )

    RE_STEP_BY_STEP = re.compile(
        r"\b(step[- ]by[- ]step|derive|derivation|mathematical\s+proof|proof\s+of|"
        r"walk\s+me\s+through|walkthrough|solve\s+step|mathematical\s+derivation|"
        r"formal\s+proof|pipeline\s+steps|workflow)\b",
        re.IGNORECASE,
    )

    RE_COMPARISON = re.compile(
        r"\b(compare|versus|vs\.?|difference\s+between|differences\s+between|"
        r"pros\s+and\s+cons|advantages\s+and\s+disadvantages|trade[- ]?offs?|"
        r"contrast|bullet\s+points|list\s+the|table\s+of|comparison)\b",
        re.IGNORECASE,
    )

    RE_CODE = re.compile(
        r"\b(code|python|script|implement|implementation|function|algorithm|"
        r"class|program|snippet|syntax|debug|refactor|write\s+a\s+program)\b",
        re.IGNORECASE,
    )

    CONVERSATIONAL_INTENTS = {
        "greeting",
        "identity",
        "farewell",
        "bot_capabilities",
        "acknowledgement",
        "gratitude",
    }

    def analyze_syntactic_cues(self, text: str) -> dict:
        """
        Uses NLTK POS tagging to inspect syntactic characteristics:
        - Verb types (imperative action vs stative)
        - Conjunction frequency (multi-clause query complexity)
        - Interrogatives (why/how vs who/when)
        """
        cues = {
            "has_imperative_derive": False,
            "has_imperative_compare": False,
            "has_imperative_code": False,
            "conjunction_count": 0,
            "is_why_or_how": False,
            "word_count": len(text.split()),
        }

        if not text or not _NLTK_AVAILABLE:
            return cues

        try:
            tokens = word_tokenize(text)
            cues["word_count"] = len(tokens)
            pos_tags = pos_tag(tokens)

            for word, tag in pos_tags:
                lower = word.lower()
                # Check conjunctions (CC = and, but, or, etc.)
                if tag == "CC":
                    cues["conjunction_count"] += 1
                # Check interrogatives
                if lower in ("why", "how"):
                    cues["is_why_or_how"] = True
                # Check verbs
                if tag in ("VB", "VBP", "VBZ", "VBG"):
                    if lower in ("derive", "calculate", "prove"):
                        cues["has_imperative_derive"] = True
                    elif lower in ("compare", "contrast", "differentiate"):
                        cues["has_imperative_compare"] = True
                    elif lower in ("implement", "code", "write", "program"):
                        cues["has_imperative_code"] = True
        except Exception as e:
            logger.debug("[TokenEstimator] POS parsing notice: %s", e)

        return cues

    def determine_response_type(
        self,
        message: str,
        intent: str = "",
        is_code: bool = False,
        fuzzy_profile: Optional[FuzzyQueryProfile] = None,
        syntactic_cues: Optional[dict] = None,
    ) -> str:
        """
        Determines the semantic response type demanded by the user query.
        Follows a strict hierarchy of specificity:
        1. Explicit ultra-concise constraint
        2. Code implementation / debugging
        3. Step-by-step mathematical derivation
        4. In-depth / comprehensive deep dive
        5. Structured comparison / multi-point breakdown
        6. Conversational / greetings / identity
        7. Standard conceptual explanation (fallback)
        """
        cues = syntactic_cues or self.analyze_syntactic_cues(message)

        # 1. Explicit user request for ultra-brevity overrides general content
        if self.RE_ULTRA_CONCISE.search(message):
            return RESPONSE_TYPE_ULTRA_CONCISE

        # 2. Curriculum roadmap / detailed syllabus request
        if any(w in message.lower() for w in ("roadmap", "curriculum", "learning path", "syllabus")) or intent == "learning_path":
            return RESPONSE_TYPE_CURRICULUM_ROADMAP

        # 3. Code implementation
        if is_code or cues.get("has_imperative_code") or (
            fuzzy_profile and fuzzy_profile.code_focus >= 0.70
        ) or (self.RE_CODE.search(message) and any(w in message.lower() for w in ("write", "show", "give", "implement", "create"))):
            return RESPONSE_TYPE_CODE_IMPLEMENTATION

        # 3. Step-by-step or mathematical derivation
        if self.RE_STEP_BY_STEP.search(message) or cues.get("has_imperative_derive") or (
            fuzzy_profile and fuzzy_profile.mathematical_detail >= 0.75 and getattr(fuzzy_profile, "explanation_depth", 0.5) >= 0.65
        ):
            return RESPONSE_TYPE_STEP_BY_STEP_DERIVATION

        # 4. In-depth / exhaustive explanation
        if self.RE_IN_DEPTH.search(message) or (
            fuzzy_profile and getattr(fuzzy_profile, "explanation_depth", 0.5) >= 0.80 and fuzzy_profile.technical_depth >= 0.70
        ):
            return RESPONSE_TYPE_IN_DEPTH_EXPLANATION

        # 5. Structured comparison / multi-point breakdown
        if self.RE_COMPARISON.search(message) or cues.get("has_imperative_compare"):
            return RESPONSE_TYPE_STRUCTURED_COMPARISON

        # 6. Pure conversational / identity / greeting
        is_continuation = bool(re.match(r"^\s*(continue|go\s+on|keep\s+going|more|tell\s+me\s+more|proceed|next)\b", message, re.IGNORECASE))
        is_direct_expl = bool(re.search(r"\b(just\s+explain|explain\s+it|tell\s+me|give\s+me\s+the\s+answer|direct\s+answer)\b", message, re.IGNORECASE))
        if not is_continuation and not is_direct_expl:
            if (
                intent in self.CONVERSATIONAL_INTENTS
                or (cues["word_count"] <= 3 and intent in ("greeting", "unknown", "identity"))
            ):
                return RESPONSE_TYPE_CONVERSATIONAL

        # 7. Standard concept explanation (default)
        return RESPONSE_TYPE_STANDARD_CONCEPT

    def estimate(
        self,
        message: str,
        intent: str = "",
        fuzzy_profile: Optional[FuzzyQueryProfile] = None,
        entities: Optional[List[EntityItem]] = None,
        is_code: bool = False,
        depth_mode: str = "auto",
    ) -> TokenBudgetDecision:
        """
        Computes calibrated token budget and response demand categorization.
        Supports explicit depth_mode ('easy', 'medium', 'hard') or 'auto' dynamic analysis.
        Uses NLP heuristics:
        - Response type determination
        - Entity count multiplier (+30 tokens per extra entity in multi-subject questions)
        - Syntactic complexity multiplier (conjunctions, clause depth)
        - Fuzzy profile modulation (technical depth, math detail, code focus)
        """
        normalized_mode = (depth_mode or "auto").lower().strip()
        if normalized_mode == "easy":
            budget = 384
            resp_type = RESPONSE_TYPE_STANDARD_CONCEPT
            demand_score = 0.35
            reason = "Selected Easy Mode (Definition / Intuition): quick conceptual definition budget (384 tokens)"
            return TokenBudgetDecision(
                token_budget=budget,
                response_type=resp_type,
                demand_score=demand_score,
                reasoning=reason,
                base_budget=budget,
                multiplier=1.0,
            )
        elif normalized_mode == "medium":
            budget = 640
            resp_type = RESPONSE_TYPE_IN_DEPTH_EXPLANATION
            demand_score = 0.65
            reason = "Selected Medium Mode (Depth Explanation): comprehensive conceptual explanation budget (640 tokens)"
            return TokenBudgetDecision(
                token_budget=budget,
                response_type=resp_type,
                demand_score=demand_score,
                reasoning=reason,
                base_budget=budget,
                multiplier=1.0,
            )
        elif normalized_mode == "hard":
            budget = 1024
            resp_type = RESPONSE_TYPE_STEP_BY_STEP_DERIVATION
            demand_score = 1.0
            reason = "Selected Hard Mode (Detailed Step-by-Step Rigor): exhaustive mathematical derivation & architecture budget (1024 tokens)"
            return TokenBudgetDecision(
                token_budget=budget,
                response_type=resp_type,
                demand_score=demand_score,
                reasoning=reason,
                base_budget=budget,
                multiplier=1.0,
            )
        elif normalized_mode == "ultra":
            budget = 2048
            resp_type = RESPONSE_TYPE_IN_DEPTH_EXPLANATION
            demand_score = 1.0
            reason = "Selected Ultra Mode (Comprehensive Masterclass): maximum deep-dive monograph budget (2048 tokens)"
            return TokenBudgetDecision(
                token_budget=budget,
                response_type=resp_type,
                demand_score=demand_score,
                reasoning=reason,
                base_budget=budget,
                multiplier=1.0,
            )

        fp = fuzzy_profile or FuzzyQueryProfile()
        entities_list = entities or []
        syntactic_cues = self.analyze_syntactic_cues(message)

        # 1. Determine Response Type
        resp_type = self.determine_response_type(
            message=message,
            intent=intent,
            is_code=is_code,
            fuzzy_profile=fp,
            syntactic_cues=syntactic_cues,
        )

        base_budget = CATEGORY_BASE_BUDGETS.get(resp_type, 200)
        min_bound, max_bound = CATEGORY_BOUNDS.get(resp_type, (160, 256))

        # 2. Compute Multipliers based on NLP features
        multiplier = 1.0
        reasons = [f"Classified as '{resp_type}' (base: {base_budget} tokens)"]

        # Entity density effect: questions comparing multiple entities need room for each
        num_entities = len(entities_list)
        if num_entities > 1 and resp_type not in (RESPONSE_TYPE_ULTRA_CONCISE, RESPONSE_TYPE_CONVERSATIONAL):
            entity_factor = min(0.25, 0.08 * (num_entities - 1))
            multiplier += entity_factor
            reasons.append(f"+{int(entity_factor * 100)}% for {num_entities} extracted entities")

        # Syntactic complexity: conjunctions indicate multi-part queries
        conjunctions = syntactic_cues.get("conjunction_count", 0)
        if conjunctions >= 2 and resp_type not in (RESPONSE_TYPE_ULTRA_CONCISE, RESPONSE_TYPE_CONVERSATIONAL):
            conj_factor = min(0.20, 0.05 * conjunctions)
            multiplier += conj_factor
            reasons.append(f"+{int(conj_factor * 100)}% for multi-clause conjunctions ({conjunctions})")

        # Mamdani Fuzzy profile modulation
        if resp_type not in (RESPONSE_TYPE_ULTRA_CONCISE, RESPONSE_TYPE_CONVERSATIONAL):
            if fp.technical_depth >= 0.75:
                multiplier += 0.10
                reasons.append("+10% high technical depth")
            elif fp.beginner_friendliness >= 0.75 or fp.technical_depth <= 0.30:
                multiplier -= 0.10
                reasons.append("-10% beginner-friendly simplified intuition")
            if fp.mathematical_detail >= 0.70:
                multiplier += 0.12
                reasons.append("+12% mathematical formulation")
            if fp.code_focus >= 0.70 or is_code:
                multiplier += 0.15
                reasons.append("+15% code implementation")

        # 3. Calculate final token limit and clamp
        raw_budget = int(round(base_budget * multiplier))
        clamped_budget = max(min_bound, min(raw_budget, max_bound))

        # 4. Continuous demand score (0.0 to 1.0)
        demand_score = round(min(1.0, max(0.0, (clamped_budget - 48) / (576 - 48))), 3)

        reasoning_str = "; ".join(reasons)

        logger.info(
            "[TokenEstimator] type=%s budget=%d (base=%d, mult=%.2f, demand=%.2f) reason=%s",
            resp_type,
            clamped_budget,
            base_budget,
            multiplier,
            demand_score,
            reasoning_str,
        )

        return TokenBudgetDecision(
            token_budget=clamped_budget,
            response_type=resp_type,
            demand_score=demand_score,
            reasoning=reasoning_str,
            base_budget=base_budget,
            multiplier=round(multiplier, 2),
        )


# Singleton instance
token_budget_estimator = TokenBudgetEstimator()
