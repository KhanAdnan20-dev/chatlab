"""Mamdani Fuzzy Inference System for Dynamic Prompt Orchestration."""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.app.config import settings
from backend.app.schemas import EntityItem, FuzzyQueryProfile
from backend.app.services.preprocessing import PreprocessedInput

logger = logging.getLogger(__name__)


def trimf(x: float | np.ndarray, params: list[float]) -> float | np.ndarray:
    """Triangular membership function. params = [a, b, c] where a <= b <= c."""
    a, b, c = params
    if isinstance(x, (int, float)):
        if x < a or x > c:
            return 0.0
        elif a <= x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:
            return (c - x) / (c - b) if c != b else 1.0
    else:
        y = np.zeros_like(x, dtype=float)
        if b > a:
            mask_left = (x >= a) & (x <= b)
            y[mask_left] = (x[mask_left] - a) / (b - a)
        if c > b:
            mask_right = (x > b) & (x <= c)
            y[mask_right] = (c - x[mask_right]) / (c - b)
        y[x == b] = 1.0
        return y


def trapmf(x: float | np.ndarray, params: list[float]) -> float | np.ndarray:
    """Trapezoidal membership function. params = [a, b, c, d] where a <= b <= c <= d."""
    a, b, c, d = params
    if isinstance(x, (int, float)):
        if x < a or x > d:
            return 0.0
        elif a <= x < b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b <= x <= c:
            return 1.0
        else:
            return (d - x) / (d - c) if d != c else 1.0
    else:
        y = np.zeros_like(x, dtype=float)
        if b > a:
            mask_left = (x >= a) & (x < b)
            y[mask_left] = (x[mask_left] - a) / (b - a)
        mask_mid = (x >= b) & (x <= c)
        y[mask_mid] = 1.0
        if d > c:
            mask_right = (x > c) & (x <= d)
            y[mask_right] = (d - x[mask_right]) / (d - c)
        return y


def defuzz_centroid(universe: np.ndarray, membership: np.ndarray, default_val: float = 0.5) -> float:
    """Centroid defuzzification of aggregated fuzzy output set."""
    area = np.sum(membership)
    if area <= 1e-9:
        return default_val
    centroid = np.sum(universe * membership) / area
    return float(np.clip(centroid, 0.0, 1.0))


@dataclass
class FuzzyInputs:
    """Crisp antecedent values fed into the Mamdani Fuzzy System."""

    confidence: float
    similarity: float
    complexity: float
    entity_density: float
    conversationalness: float
    is_code: bool = False
    has_context: bool = False


@dataclass
class FuzzyRouterDecision:
    """Structured decision output from the Mamdani Fuzzy Prompt Orchestrator."""

    selected_layer: str
    activation_score: float
    confidence: float
    similarity: float
    complexity: float
    entity_density: float
    conversationalness: float
    fuzzy_profile: FuzzyQueryProfile
    reason: str
    is_fallback: bool
    fired_rules: list[str] = field(default_factory=list)


class MamdaniFuzzyRouter:
    """
    Genuine Mamdani Fuzzy Inference System for Dynamic Prompt Orchestration.
    Evaluates membership functions across NLP antecedents, performs Mamdani
    min-implication and max-aggregation, and applies centroid defuzzification
    to generate a 10-dimension Fuzzy Query Profile that guides Gemma 4.
    """

    CONVERSATIONAL_KEYWORDS = [
        r"\bhow\b", r"\bwhy\b", r"\bshould\b", r"\broadmap\b", r"\bintuitively\b",
        r"\bexplain\b", r"\bcompare\b", r"\bbetter\b", r"\bthink\b", r"\bnext\b",
        r"\bdeep\b", r"\bstep[- ]by[- ]step\b", r"\bbest\b", r"\brecommend\b",
        r"\badvice\b", r"\blearn\b", r"\bguide\b", r"\bpath\b", r"\bversus\b",
        r"\bdifference\b", r"\buseful\b", r"\bimbalanced\b", r"\bwhen\b",
        r"\blike\s+i'?m\s+5\b", r"\bteach\s+me\b"
    ]

    MATH_KEYWORDS = [
        r"\bformula\b", r"\bequation\b", r"\bmath\b", r"\bloss\s+function\b",
        r"\bderivative\b", r"\bgradient\b", r"\bcalculate\b", r"\bf1\s+formula\b",
        r"\bmatrix\b", r"\bproof\b", r"\bprecision\b", r"\brecall\b", r"\bloocv\b",
        r"\bcalculus\b", r"\bprobability\b", r"\bbayes\b", r"\blinear\s+algebra\b",
        r"\bdot\s+product\b", r"\beigen\w*\b", r"\bsoftmax\b", r"\bbackprop\w*\b",
        r"\bcost\s+function\b", r"\bobjective\s+function\b", r"\bvector\b",
        r"\badmissib\w*\b", r"\bconsisten\w*\b", r"\bcentroid\b", r"\bdefuzz\w*\b",
        r"\bmembership\s+function\b", r"\bunification\b", r"\bresolution\b"
    ]

    AIML_SYLLABUS_PATTERN = re.compile(
        r"(?:\b(?:ai|ml|machine\s+learning|deep\s+learning|soft\s+computing|hard\s+computing|"
        r"zadeh|fuzzy|mamdani|sugeno|centroid|defuzz\w*|crisp|linguistic|trapezoidal|triangular|"
        r"neural|perceptron|mlp|backprop\w*|kohonen|som|anfis|jang|"
        r"search|bfs|dfs|greedy|heuristic|admissib\w*|state[- ]space|"
        r"first[- ]order\s+logic|fol|unification|resolution|strips|planning|"
        r"genetic\s+algorithm|chromosome|crossover|mutation|roulette|tournament|fitness|"
        r"overfitting|loss|gradient|f1|loocv|svm|regression|precision|recall|"
        r"dataset|epoch|batch|transformer|bert|gpt|cnn|rnn|classification|clustering|"
        r"python|pytorch)\b|a\*)",
        re.IGNORECASE,
    )

    GENERAL_CHITCHAT_PATTERN = re.compile(
        r"^\s*(?:hello|hi|hey|who\s+are\s+you|what\s+can\s+you\s+do|thanks|thank\s+you|"
        r"bye|goodbye|how\s+are\s+you|weather|2\s*\+\s*2|what\s+is\s+2\s*\+\s*2)[\s\?!\.]*$",
        re.IGNORECASE,
    )

    EXAMPLE_KEYWORDS = [
        r"\bexample\b", r"\banalogy\b", r"\bsample\b", r"\bshow\s+me\b",
        r"\bintuitively\b", r"\blike\s+i'?m\s+5\b", r"\bhow\s+does\s+it\s+work\b",
        r"\bscenario\b", r"\buse\s+case\b"
    ]

    CODE_KEYWORDS = [
        r"\bcode\b", r"\bpython\b", r"\bpytorch\b", r"\bdebug\b", r"\bscript\b",
        r"\bfunction\b", r"\bclass\b", r"\bdef\b", r"\bimport\b", r"\btensor\b",
        r"\bimplementation\b", r"\balgorithm\b", r"\bsyntax\b"
    ]

    BEGINNER_KEYWORDS = [
        r"\blike\s+i'?m\s+5\b", r"\bfive\b", r"\bsimpl(e|y)\b", r"\bintuitively\b",
        r"\bbeginner\b", r"\beasy\b", r"\bbasics?\b", r"\bteach\s+me\b",
        r"\bplain\s+english\b", r"\bfor\s+dummies\b", r"\boverview\b", r"\bhigh[- ]level\b",
        r"\bwhat\s+is\b", r"\bdefine\b"
    ]

    DEEP_KEYWORDS = [
        r"\bdeep[- ]dive\b", r"\bstep[- ]by[- ]step\b", r"\bin[- ]depth\b", r"\bin[- ]detail\b",
        r"\bthorough\w*\b", r"\barchitecture\b", r"\bderivation\b", r"\bderive\b",
        r"\bunder\s+the\s+hood\b", r"\binternals\b", r"\brigor\w*\b", r"\bexhaustive\b",
        r"\bformal\s+proof\b", r"\bmathematical\b", r"\btheoretical\b"
    ]

    def __init__(self) -> None:
        self.output_universe = np.linspace(0.0, 1.0, 101)

        # Antecedent membership function parameters
        self.conf_low_params = [0.0, 0.0, 0.25, 0.45]
        self.conf_med_params = [0.30, 0.55, 0.75]
        self.conf_high_params = [0.60, 0.80, 1.0, 1.0]

        self.sim_low_params = [0.0, 0.0, 0.20, 0.40]
        self.sim_med_params = [0.25, 0.50, 0.70]
        self.sim_high_params = [0.55, 0.75, 1.0, 1.0]

        self.comp_simple_params = [0.0, 0.0, 0.20, 0.40]
        self.comp_mod_params = [0.25, 0.50, 0.70]
        self.comp_complex_params = [0.55, 0.75, 1.0, 1.0]

        self.ent_low_params = [0.0, 0.0, 0.15, 0.35]
        self.ent_med_params = [0.20, 0.45, 0.70]
        self.ent_high_params = [0.50, 0.75, 1.0, 1.0]

        self.conv_low_params = [0.0, 0.0, 0.20, 0.40]
        self.conv_med_params = [0.25, 0.50, 0.70]
        self.conv_high_params = [0.55, 0.75, 1.0, 1.0]

        # Output universe linguistic membership functions
        self.out_low = trapmf(self.output_universe, [0.0, 0.0, 0.20, 0.40])
        self.out_med = trimf(self.output_universe, [0.25, 0.50, 0.75])
        self.out_high = trapmf(self.output_universe, [0.60, 0.80, 1.0, 1.0])

    def compute_conversationalness(self, text: str, intent: str, confidence: float) -> float:
        if not text or not text.strip():
            return 0.0

        t_lower = text.lower()
        hits = sum(1 for pat in self.CONVERSATIONAL_KEYWORDS if re.search(pat, t_lower))
        words = t_lower.split()
        token_count = len(words)

        length_bonus = min(max(0, token_count - 2) / 8.0, 0.4)
        keyword_bonus = min(hits * 0.30, 0.6)

        base_conv = length_bonus + keyword_bonus
        if intent in ["greeting", "goodbye", "thanks"] and hits == 0:
            base_conv = 0.85
        elif intent == "learning_roadmap" or "roadmap" in t_lower:
            base_conv = 0.95

        return round(float(np.clip(base_conv, 0.20, 1.0)), 4)

    def compute_query_complexity(self, text: str, preprocessed: PreprocessedInput) -> float:
        if not text or not text.strip():
            return 0.0

        tokens = preprocessed.tokens
        num_tokens = len(tokens)
        num_chars = len(text.strip())

        if num_tokens == 0:
            return 0.0

        lexical_diversity = len(set(tokens)) / num_tokens
        token_factor = min(num_tokens / 18.0, 1.0)
        char_factor = min(num_chars / 100.0, 1.0)

        complexity = 0.50 * token_factor + 0.35 * char_factor + 0.15 * lexical_diversity
        return round(float(np.clip(complexity, 0.0, 1.0)), 4)

    def compute_entity_density(
        self,
        entities: list[EntityItem],
        preprocessed: PreprocessedInput,
    ) -> float:
        num_tokens = len(preprocessed.tokens)
        if num_tokens == 0 or not entities:
            return 0.0

        raw_ratio = len(entities) / max(num_tokens, 1)
        normalized_density = min(raw_ratio * 2.2, 1.0)
        return round(float(np.clip(normalized_density, 0.0, 1.0)), 4)

    def _infer_characteristic(
        self,
        rules: List[Tuple[float, str]],
        default_val: float = 0.5,
    ) -> float:
        """
        Mamdani min-implication and max-aggregation over linguistic rules.
        Each rule is (firing_strength, target_linguistic_level in {'low', 'med', 'high'}).
        """
        agg = np.zeros_like(self.output_universe)
        for w, target in rules:
            if w <= 1e-6:
                continue
            if target == "low":
                out_shape = self.out_low
            elif target == "med":
                out_shape = self.out_med
            else:
                out_shape = self.out_high

            # Min implication
            clipped = np.minimum(w, out_shape)
            # Max aggregation
            agg = np.maximum(agg, clipped)

        return round(defuzz_centroid(self.output_universe, agg, default_val=default_val), 2)

    def generate_fuzzy_profile(
        self,
        text: str,
        intent: str,
        confidence: float,
        similarity: float,
        complexity: float,
        entity_density: float,
        conversationalness: float,
        is_code: bool = False,
        has_context: bool = False,
        depth_mode: str = "auto",
    ) -> Tuple[FuzzyQueryProfile, List[str]]:
        """
        Executes genuine Mamdani Fuzzy Inference:
        Fuzzifies inputs -> Evaluates rules -> Aggregates sets -> Centroid defuzzification.
        Supports explicit user depth_mode override ('easy', 'medium', 'hard') or 'auto' FIS.
        """
        t_lower = text.lower() if text else ""

        # Explicit user depth_mode overrides
        normalized_mode = (depth_mode or "auto").lower().strip()
        if normalized_mode == "easy":
            fired_rules_log = [
                "Depth Mode: Easy (Definition / Beginner-Focused)",
                "Mamdani Forced Profile: Low Technical Depth (0.20) for rapid definition & simplicity",
                "Mamdani Forced Profile: High Beginner Friendliness (0.95) & High Example Preference (0.85)",
                "Mamdani Forced Profile: Minimized Mathematical Formulations (0.15)",
            ]
            profile = FuzzyQueryProfile(
                technical_depth=0.20,
                beginner_friendliness=0.95,
                conversationalness=max(conversationalness, 0.70),
                mathematical_detail=0.15,
                example_preference=0.85,
                code_focus=0.75 if is_code else 0.10,
                aiml_relevance=0.80,
                explanation_depth=0.35,
                query_complexity=complexity,
                ambiguity_level=0.10,
            )
            return profile, fired_rules_log

        elif normalized_mode == "medium":
            fired_rules_log = [
                "Depth Mode: Medium (Balanced In-Depth Explanation)",
                "Mamdani Profile: Moderate Technical Depth (0.65) for conceptual thoroughness",
                "Mamdani Profile: Balanced Beginner Friendliness (0.50) & Practical Examples (0.65)",
                "Mamdani Profile: Moderate Mathematical Formulations (0.50)",
            ]
            profile = FuzzyQueryProfile(
                technical_depth=0.65,
                beginner_friendliness=0.50,
                conversationalness=conversationalness,
                mathematical_detail=0.50,
                example_preference=0.65,
                code_focus=0.80 if is_code else 0.20,
                aiml_relevance=0.85,
                explanation_depth=0.65,
                query_complexity=complexity,
                ambiguity_level=0.15,
            )
            return profile, fired_rules_log

        elif normalized_mode == "hard":
            fired_rules_log = [
                "Depth Mode: Hard (Extremely Detailed Step-by-Step Rigor)",
                "Mamdani Profile: Maximum Technical Depth (0.95) for formal academic rigor",
                "Mamdani Profile: Exhaustive Explanation Depth (0.95) & Step-by-Step Breakdown",
                "Mamdani Profile: Rigorous Mathematical Formulation (0.90) with LaTeX",
            ]
            profile = FuzzyQueryProfile(
                technical_depth=0.95,
                beginner_friendliness=0.15,
                conversationalness=conversationalness,
                mathematical_detail=0.90,
                example_preference=0.40,
                code_focus=0.85 if is_code else 0.25,
                aiml_relevance=0.95,
                explanation_depth=0.95,
                query_complexity=max(complexity, 0.85),
                ambiguity_level=0.10,
            )
            return profile, fired_rules_log

        elif normalized_mode == "ultra":
            fired_rules_log = [
                "Depth Mode: Ultra (Comprehensive Masterclass / Complete Monograph)",
                "Mamdani Profile: Maximum Technical Depth (1.00) & Formal Mathematical Derivations",
                "Mamdani Profile: Complete Explanation Depth (1.00) & Step-by-Step Architecture",
                "Mamdani Profile: Rigorous Mathematical Formulation (0.98) with Canonical LaTeX",
            ]
            profile = FuzzyQueryProfile(
                technical_depth=1.0,
                beginner_friendliness=0.10,
                conversationalness=conversationalness,
                mathematical_detail=0.98,
                example_preference=0.60,
                code_focus=0.90 if is_code else 0.35,
                aiml_relevance=1.0,
                explanation_depth=1.0,
                query_complexity=max(complexity, 0.95),
                ambiguity_level=0.05,
            )
            return profile, fired_rules_log

        # Auto Mode: Dynamic Mamdani Fuzzy Inference System
        # 1. Fuzzification of Antecedents
        mu_c_low = trapmf(confidence, self.conf_low_params)
        mu_c_med = trimf(confidence, self.conf_med_params)
        mu_c_high = trapmf(confidence, self.conf_high_params)

        mu_l_simple = trapmf(complexity, self.comp_simple_params)
        mu_l_mod = trimf(complexity, self.comp_mod_params)
        mu_l_complex = trapmf(complexity, self.comp_complex_params)

        mu_e_low = trapmf(entity_density, self.ent_low_params)
        mu_e_med = trimf(entity_density, self.ent_med_params)
        mu_e_high = trapmf(entity_density, self.ent_high_params)

        mu_k_low = trapmf(conversationalness, self.conv_low_params)
        mu_k_med = trimf(conversationalness, self.conv_med_params)
        mu_k_high = trapmf(conversationalness, self.conv_high_params)

        # Domain signals & keyword matching
        has_beginner = any(re.search(p, t_lower) for p in self.BEGINNER_KEYWORDS)
        has_deep = any(re.search(p, t_lower) for p in self.DEEP_KEYWORDS)
        has_math = any(re.search(p, t_lower) for p in self.MATH_KEYWORDS) or intent == "formula_request"
        has_example = any(re.search(p, t_lower) for p in self.EXAMPLE_KEYWORDS)
        has_code_kw = any(re.search(p, t_lower) for p in self.CODE_KEYWORDS) or is_code

        # Ambiguity input signal
        ambiguity_val = float(np.clip(1.0 - max(confidence, similarity), 0.0, 1.0))

        fired_rules_log: List[str] = []

        # --- Rule Base 1: Technical Depth ---
        # If deep keywords present OR (complexity is High and entity density is High) -> High
        # If beginner keywords present OR complexity is Simple -> Low
        r_tech_high = max(min(mu_l_complex, mu_e_high), 0.95 if has_deep else (0.80 if "deep" in t_lower or "advanced" in t_lower else 0.0))
        r_tech_low = max(min(mu_l_simple, mu_e_low), 0.90 if has_beginner else (0.75 if mu_l_simple > 0.6 else 0.0))
        r_tech_med = max(mu_l_mod, min(mu_c_high, mu_l_simple))
        tech_rules = [(r_tech_high, "high"), (r_tech_med, "med"), (r_tech_low, "low")]
        technical_depth = self._infer_characteristic(tech_rules, default_val=0.50)
        if r_tech_high > 0.4:
            fired_rules_log.append(f"Mamdani Rule [TechDepth: High] fired (w={r_tech_high:.2f})")
        elif r_tech_low > 0.4:
            fired_rules_log.append(f"Mamdani Rule [TechDepth: Low] fired (w={r_tech_low:.2f})")

        # --- Rule Base 2: Beginner Friendliness ---
        # Beginner keywords or Low technical depth -> High
        # Deep keywords or (High technical depth and Complex query) -> Low
        r_beg_high = max(0.95 if has_beginner else 0.0, trapmf(technical_depth, [0.0, 0.0, 0.25, 0.45]))
        r_beg_low = max(0.90 if has_deep else 0.0, min(mu_l_complex, mu_e_high))
        r_beg_med = 0.50
        beg_rules = [(r_beg_high, "high"), (r_beg_med, "med"), (r_beg_low, "low")]
        beginner_friendliness = self._infer_characteristic(beg_rules, default_val=0.60)
        if r_beg_high > 0.5:
            fired_rules_log.append(f"Mamdani Rule [BeginnerFriendliness: High] fired (w={r_beg_high:.2f})")

        # --- Rule Base 3: Explanation Depth ---
        # Deep query or complex query with high confidence -> In-Depth (High)
        # Simple query with low complexity -> Concise (Low)
        r_depth_high = max(0.95 if has_deep else 0.0, min(mu_l_complex, mu_c_high))
        r_depth_low = max(mu_l_simple, 0.85 if has_beginner else 0.0, 0.85 if len(t_lower.split()) <= 3 else 0.0)
        r_depth_med = mu_l_mod
        depth_rules = [(r_depth_high, "high"), (r_depth_med, "med"), (r_depth_low, "low")]
        explanation_depth = self._infer_characteristic(depth_rules, default_val=0.50)
        if r_depth_high > 0.4:
            fired_rules_log.append(f"Mamdani Rule [ExplanationDepth: High] fired (w={r_depth_high:.2f})")
        elif r_depth_low > 0.5:
            fired_rules_log.append(f"Mamdani Rule [ExplanationDepth: Concise] fired (w={r_depth_low:.2f})")

        # --- Rule Base 4: Mathematical Detail ---
        r_math_high = 0.95 if has_math else min(mu_e_high, mu_l_complex)
        r_math_low = 0.90 if not has_math and (has_beginner or mu_l_simple > 0.5) else 0.30
        r_math_med = 0.40 if not has_math else 0.0
        math_rules = [(r_math_high, "high"), (r_math_med, "med"), (r_math_low, "low")]
        mathematical_detail = self._infer_characteristic(math_rules, default_val=0.25)
        if r_math_high > 0.5:
            fired_rules_log.append(f"Mamdani Rule [MathDetail: High] fired (w={r_math_high:.2f})")

        # --- Rule Base 5: Example Preference ---
        r_ex_high = max(0.92 if has_example or has_beginner else 0.0, mu_k_high)
        r_ex_med = 0.50
        r_ex_low = 0.20 if not has_example and mu_l_complex > 0.7 else 0.0
        ex_rules = [(r_ex_high, "high"), (r_ex_med, "med"), (r_ex_low, "low")]
        example_preference = self._infer_characteristic(ex_rules, default_val=0.65)

        # --- Rule Base 6: Code Focus ---
        r_code_high = 1.0 if is_code else (0.85 if has_code_kw else 0.0)
        r_code_low = 0.90 if not is_code and not has_code_kw else 0.0
        r_code_med = 0.40 if has_code_kw and not is_code else 0.0
        code_rules = [(r_code_high, "high"), (r_code_med, "med"), (r_code_low, "low")]
        code_focus = self._infer_characteristic(code_rules, default_val=0.10)
        if r_code_high > 0.5:
            fired_rules_log.append(f"Mamdani Rule [CodeFocus: High] fired (w={r_code_high:.2f})")

        # --- Rule Base 7: Conversationalness ---
        r_conv_high = max(mu_k_high, 0.90 if has_context else 0.0)
        r_conv_med = mu_k_med
        r_conv_low = mu_k_low
        conv_rules = [(r_conv_high, "high"), (r_conv_med, "med"), (r_conv_low, "low")]
        conversational_style = self._infer_characteristic(conv_rules, default_val=0.60)

        # --- Rule Base 8: AIML Relevance ---
        is_aiml_explicit = (
            bool(self.AIML_SYLLABUS_PATTERN.search(t_lower))
            or is_code
            or similarity >= 0.40
        )

        is_purely_general = (
            bool(self.GENERAL_CHITCHAT_PATTERN.match(t_lower))
            and not is_aiml_explicit
        )

        r_aiml_high = 0.95 if is_aiml_explicit else (0.20 if is_purely_general else 0.60)
        r_aiml_low = 0.90 if is_purely_general else 0.0
        r_aiml_med = 0.50 if not is_aiml_explicit and not is_purely_general else 0.0
        aiml_rules = [(r_aiml_high, "high"), (r_aiml_med, "med"), (r_aiml_low, "low")]
        aiml_relevance = self._infer_characteristic(aiml_rules, default_val=0.75)

        # --- Rule Base 9: Ambiguity Level ---
        r_amb_high = trapmf(ambiguity_val, [0.55, 0.75, 1.0, 1.0])
        r_amb_med = trimf(ambiguity_val, [0.25, 0.50, 0.75])
        r_amb_low = trapmf(ambiguity_val, [0.0, 0.0, 0.25, 0.45])
        amb_rules = [(r_amb_high, "high"), (r_amb_med, "med"), (r_amb_low, "low")]
        ambiguity_level = self._infer_characteristic(amb_rules, default_val=0.20)

        profile = FuzzyQueryProfile(
            technical_depth=technical_depth,
            beginner_friendliness=beginner_friendliness,
            conversationalness=conversational_style,
            mathematical_detail=mathematical_detail,
            example_preference=example_preference,
            code_focus=code_focus,
            aiml_relevance=aiml_relevance,
            explanation_depth=explanation_depth,
            query_complexity=complexity,
            ambiguity_level=ambiguity_level,
        )

        return profile, fired_rules_log

    def route(
        self,
        confidence: float,
        similarity: float,
        complexity: float,
        entity_density: float,
        conversationalness: float = 0.0,
        intent: str = "unknown",
        text: str = "",
        is_code: bool = False,
        has_context: bool = False,
        depth_mode: str = "auto",
    ) -> FuzzyRouterDecision:
        """
        Derives Fuzzy Query Profile via Mamdani FIS for Gemma 4 prompt orchestration.
        Supports depth_mode: 'auto', 'easy', 'medium', 'hard'.
        """
        effective_conv = conversationalness
        if effective_conv == 0.0 and text:
            effective_conv = self.compute_conversationalness(text, intent, confidence)

        fuzzy_profile, fired_rules = self.generate_fuzzy_profile(
            text=text,
            intent=intent,
            confidence=confidence,
            similarity=similarity,
            complexity=complexity,
            entity_density=entity_density,
            conversationalness=effective_conv,
            is_code=is_code,
            has_context=has_context,
            depth_mode=depth_mode,
        )

        activation_score = round(
            float(
                0.25 * fuzzy_profile.conversationalness +
                0.25 * fuzzy_profile.technical_depth +
                0.25 * fuzzy_profile.aiml_relevance +
                0.25 * fuzzy_profile.code_focus
            ),
            4,
        )

        summary_rules = [
            f"Mamdani FIS -> Tech Depth: {fuzzy_profile.technical_depth:.2f}, Beginner: {fuzzy_profile.beginner_friendliness:.2f}",
            f"Mamdani FIS -> Math: {fuzzy_profile.mathematical_detail:.2f}, Example: {fuzzy_profile.example_preference:.2f}, Code: {fuzzy_profile.code_focus:.2f}",
            f"Mamdani FIS -> Conversational: {fuzzy_profile.conversationalness:.2f}, AIML: {fuzzy_profile.aiml_relevance:.2f}"
        ]

        # In unified architecture, Gemma (Layer 2) is the primary engine
        selected_layer = "layer2"

        return FuzzyRouterDecision(
            selected_layer=selected_layer,
            activation_score=activation_score,
            confidence=confidence,
            similarity=similarity,
            complexity=complexity,
            entity_density=entity_density,
            conversationalness=effective_conv,
            fuzzy_profile=fuzzy_profile,
            reason="Mamdani Fuzzy Query Profile derived for Gemma 4 E2B Instruct prompt orchestration.",
            is_fallback=False,
            fired_rules=summary_rules + fired_rules,
        )


fuzzy_router = MamdaniFuzzyRouter()
