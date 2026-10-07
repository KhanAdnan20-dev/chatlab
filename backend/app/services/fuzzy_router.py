from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from backend.app.config import settings
from backend.app.schemas import EntityItem
from backend.app.services.preprocessing import PreprocessedInput


def trimf(x: float | np.ndarray, params: list[float]) -> float | np.ndarray:
    """
    Triangular membership function.
    params = [a, b, c] where a <= b <= c
    """
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
        # Left slope
        if b > a:
            mask_left = (x >= a) & (x <= b)
            y[mask_left] = (x[mask_left] - a) / (b - a)
        # Right slope
        if c > b:
            mask_right = (x > b) & (x <= c)
            y[mask_right] = (c - x[mask_right]) / (c - b)
        y[x == b] = 1.0
        return y


def trapmf(x: float | np.ndarray, params: list[float]) -> float | np.ndarray:
    """
    Trapezoidal membership function.
    params = [a, b, c, d] where a <= b <= c <= d
    """
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
        # Left slope
        if b > a:
            mask_left = (x >= a) & (x < b)
            y[mask_left] = (x[mask_left] - a) / (b - a)
        # Flat top
        mask_mid = (x >= b) & (x <= c)
        y[mask_mid] = 1.0
        # Right slope
        if d > c:
            mask_right = (x > c) & (x <= d)
            y[mask_right] = (d - x[mask_right]) / (d - c)
        return y


@dataclass
class FuzzyInputs:
    """Crisp values fed into the Fuzzy Inference System."""

    confidence: float
    similarity: float
    complexity: float
    entity_density: float


@dataclass
class FuzzyRouterDecision:
    """Structured decision output from the Fuzzy Inference System."""

    selected_layer: str
    activation_score: float
    confidence: float
    similarity: float
    complexity: float
    entity_density: float
    reason: str
    is_fallback: bool
    fired_rules: list[str] = field(default_factory=list)


class MamdaniFuzzyRouter:
    """
    Mamdani Fuzzy Inference System for Soft Computing Decision Routing.
    Arbitrates query dispatching between Layer 1 (Deterministic/Statistical NLP)
    and Layer 2 (Generative LLM) based on multi-criteria uncertainty analysis.
    """

    def __init__(self) -> None:
        # Discretized universe for defuzzification in [0, 1]
        self.output_universe = np.linspace(0.0, 1.0, 101)

        # Output Membership Functions: Layer Activation Score A in [0, 1]
        # Low score = Layer 1; High score = Layer 2
        self.out_layer1_params = [0.0, 0.0, 0.20, 0.40]
        self.out_hybrid_params = [0.25, 0.50, 0.75]
        self.out_layer2_params = [0.60, 0.80, 1.0, 1.0]

        # Input 1: Intent Confidence C in [0, 1]
        self.conf_low_params = [0.0, 0.0, 0.25, 0.45]
        self.conf_med_params = [0.30, 0.55, 0.75]
        self.conf_high_params = [0.60, 0.80, 1.0, 1.0]

        # Input 2: Semantic Similarity S in [0, 1]
        self.sim_low_params = [0.0, 0.0, 0.20, 0.40]
        self.sim_med_params = [0.25, 0.50, 0.70]
        self.sim_high_params = [0.55, 0.75, 1.0, 1.0]

        # Input 3: Query Complexity L in [0, 1]
        self.comp_simple_params = [0.0, 0.0, 0.20, 0.40]
        self.comp_mod_params = [0.25, 0.50, 0.70]
        self.comp_complex_params = [0.55, 0.75, 1.0, 1.0]

        # Input 4: Entity Density E in [0, 1]
        self.ent_low_params = [0.0, 0.0, 0.15, 0.35]
        self.ent_med_params = [0.20, 0.45, 0.70]
        self.ent_high_params = [0.50, 0.75, 1.0, 1.0]

    def compute_query_complexity(self, text: str, preprocessed: PreprocessedInput) -> float:
        """
        Derives a normalized complexity score L in [0, 1] based on token count,
        character length, and lexical diversity.
        """
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
        """
        Calculates normalized entity density E in [0, 1] as the ratio of
        recognized entities relative to useful tokens.
        """
        num_tokens = len(preprocessed.tokens)
        if num_tokens == 0 or not entities:
            return 0.0

        raw_ratio = len(entities) / max(num_tokens, 1)
        normalized_density = min(raw_ratio * 2.2, 1.0)
        return round(float(np.clip(normalized_density, 0.0, 1.0)), 4)

    def fuzzify_inputs(self, inputs: FuzzyInputs) -> dict[str, dict[str, float]]:
        """Evaluates degree of membership for all input variables across their linguistic terms."""
        c, s, l, e = inputs.confidence, inputs.similarity, inputs.complexity, inputs.entity_density

        memberships = {
            "confidence": {
                "LOW": float(trapmf(c, self.conf_low_params)),
                "MEDIUM": float(trimf(c, self.conf_med_params)),
                "HIGH": float(trapmf(c, self.conf_high_params)),
            },
            "similarity": {
                "LOW": float(trapmf(s, self.sim_low_params)),
                "MEDIUM": float(trimf(s, self.sim_med_params)),
                "HIGH": float(trapmf(s, self.sim_high_params)),
            },
            "complexity": {
                "SIMPLE": float(trapmf(l, self.comp_simple_params)),
                "MODERATE": float(trimf(l, self.comp_mod_params)),
                "COMPLEX": float(trapmf(l, self.comp_complex_params)),
            },
            "entity_density": {
                "LOW": float(trapmf(e, self.ent_low_params)),
                "MEDIUM": float(trimf(e, self.ent_med_params)),
                "HIGH": float(trapmf(e, self.ent_high_params)),
            },
        }
        return memberships

    def evaluate_rule_base(self, mu: dict[str, dict[str, float]]) -> tuple[dict[str, float], list[str]]:
        """
        Evaluates the 18 Mamdani fuzzy rules using Min (AND) T-norm.
        Returns aggregated consequent strengths and a trace of fired rules.
        """
        consequent_strengths = {"LAYER1": 0.0, "HYBRID": 0.0, "LAYER2": 0.0}
        fired_rules: list[str] = []

        def apply_rule(rule_id: str, antecedent_val: float, consequent_set: str, desc: str):
            if antecedent_val > 0.001:
                consequent_strengths[consequent_set] = max(
                    consequent_strengths[consequent_set], antecedent_val
                )
                fired_rules.append(f"R{rule_id} ({desc}): firing={antecedent_val:.3f} -> {consequent_set}")

        c_low = mu["confidence"]["LOW"]
        c_med = mu["confidence"]["MEDIUM"]
        c_high = mu["confidence"]["HIGH"]

        s_low = mu["similarity"]["LOW"]
        s_med = mu["similarity"]["MEDIUM"]
        s_high = mu["similarity"]["HIGH"]

        l_simp = mu["complexity"]["SIMPLE"]
        l_mod = mu["complexity"]["MODERATE"]
        l_comp = mu["complexity"]["COMPLEX"]

        e_low = mu["entity_density"]["LOW"]
        e_med = mu["entity_density"]["MEDIUM"]
        e_high = mu["entity_density"]["HIGH"]

        # RULE 1: Standard high-confidence greeting/query -> LAYER1
        apply_rule("1", min(c_high, s_high, l_simp), "LAYER1", "Conf HIGH & Sim HIGH & Comp SIMPLE")

        # RULE 2: High confidence with moderate complexity -> LAYER1
        apply_rule("2", min(c_high, s_high, l_mod), "LAYER1", "Conf HIGH & Sim HIGH & Comp MODERATE")

        # RULE 3: Low confidence, low similarity, complex query -> LAYER2
        apply_rule("3", min(c_low, s_low, l_comp), "LAYER2", "Conf LOW & Sim LOW & Comp COMPLEX")

        # RULE 4: Low confidence with moderate similarity but complex query -> LAYER2
        apply_rule("4", min(c_low, s_med, l_comp), "LAYER2", "Conf LOW & Sim MED & Comp COMPLEX")

        # RULE 5: Medium confidence, medium similarity, moderate complexity -> HYBRID
        apply_rule("5", min(c_med, s_med, l_mod), "HYBRID", "Conf MED & Sim MED & Comp MODERATE")

        # RULE 6: High confidence, but very complex query -> HYBRID
        apply_rule("6", min(c_high, l_comp), "HYBRID", "Conf HIGH & Comp COMPLEX")

        # RULE 7: Low confidence but rescued by high semantic similarity in simple query -> LAYER1
        apply_rule("7", min(c_low, s_high, l_simp), "LAYER1", "Conf LOW & Sim HIGH & Comp SIMPLE")

        # RULE 8: High confidence, high entity density, simple query -> LAYER1
        apply_rule("8", min(c_high, e_high, l_simp), "LAYER1", "Conf HIGH & Ent HIGH & Comp SIMPLE")

        # RULE 9: Medium confidence and high similarity -> LAYER1
        apply_rule("9", min(c_med, s_high, l_simp), "LAYER1", "Conf MED & Sim HIGH & Comp SIMPLE")

        # RULE 10: Low confidence, low similarity, simple/mod query -> LAYER2
        apply_rule("10", min(c_low, s_low), "LAYER2", "Conf LOW & Sim LOW")

        # RULE 11: Low confidence, medium similarity, low entity density -> LAYER2
        apply_rule("11", min(c_low, s_med, e_low), "LAYER2", "Conf LOW & Sim MED & Ent LOW")

        # RULE 12: High confidence, low similarity, simple query -> LAYER1
        apply_rule("12", min(c_high, s_low, l_simp), "LAYER1", "Conf HIGH & Sim LOW & Comp SIMPLE")

        # RULE 13: Medium confidence, high similarity, moderate complexity -> LAYER1
        apply_rule("13", min(c_med, s_high, l_mod), "LAYER1", "Conf MED & Sim HIGH & Comp MODERATE")

        # RULE 14: Moderate complexity, high entity density, medium confidence -> HYBRID
        apply_rule("14", min(l_mod, e_high, c_med), "HYBRID", "Comp MOD & Ent HIGH & Conf MED")

        # RULE 15: Complex query with high entity density and low confidence -> LAYER2
        apply_rule("15", min(l_comp, e_high, c_low), "LAYER2", "Comp COMPLEX & Ent HIGH & Conf LOW")

        # RULE 16: High confidence, moderate similarity, low complexity -> LAYER1
        apply_rule("16", min(c_high, s_med, l_simp), "LAYER1", "Conf HIGH & Sim MED & Comp SIMPLE")

        # RULE 17: Medium confidence, low similarity, moderate complexity -> HYBRID
        apply_rule("17", min(c_med, s_low, l_mod), "HYBRID", "Conf MED & Sim LOW & Comp MODERATE")

        # RULE 18: Complex query, moderate similarity -> LAYER2
        apply_rule("18", min(l_comp, s_med), "LAYER2", "Comp COMPLEX & Sim MED")

        return consequent_strengths, fired_rules

    def defuzzify_centroid(self, consequent_strengths: dict[str, float]) -> float:
        """
        Aggregates output fuzzy sets using Max and defuzzifies via Centroid (Center of Gravity).
        """
        z = self.output_universe
        mu_l1 = trapmf(z, self.out_layer1_params)
        mu_hyb = trimf(z, self.out_hybrid_params)
        mu_l2 = trapmf(z, self.out_layer2_params)

        clipped_l1 = np.minimum(consequent_strengths["LAYER1"], mu_l1)
        clipped_hyb = np.minimum(consequent_strengths["HYBRID"], mu_hyb)
        clipped_l2 = np.minimum(consequent_strengths["LAYER2"], mu_l2)

        aggregated = np.maximum(clipped_l1, np.maximum(clipped_hyb, clipped_l2))

        total_area = np.sum(aggregated)
        if total_area == 0:
            return 0.50

        centroid = np.sum(z * aggregated) / total_area
        return round(float(centroid), 4)

    def route(
        self,
        confidence: float,
        similarity: float,
        complexity: float,
        entity_density: float,
        intent: str = "unknown",
    ) -> FuzzyRouterDecision:
        """
        Executes complete Mamdani Fuzzy Inference routing pipeline:
        Fuzzification -> Rule Evaluation -> Aggregation -> Defuzzification -> Crisp Decision.
        """
        inputs = FuzzyInputs(
            confidence=confidence,
            similarity=similarity,
            complexity=complexity,
            entity_density=entity_density,
        )

        memberships = self.fuzzify_inputs(inputs)
        strengths, fired_rules = self.evaluate_rule_base(memberships)
        activation_score = self.defuzzify_centroid(strengths)

        # If intent is explicitly 'unknown' (out of domain), it always requires Layer 2
        if intent == "unknown":
            selected_layer = "layer2_pending"
            is_fallback = True
            reason = (
                f"Intent is 'unknown' (out-of-domain) with activation score {activation_score:.2f}, "
                f"routing to Layer 2."
            )
        elif activation_score <= 0.50:
            selected_layer = "layer1"
            is_fallback = False
            reason = (
                f"Fuzzy activation score ({activation_score:.2f} <= 0.50) confirms Layer 1 "
                f"suitability (Confidence: {confidence:.2f}, Similarity: {similarity:.2f}, Complexity: {complexity:.2f})."
            )
        else:
            selected_layer = "layer2_pending"
            is_fallback = True
            reason = (
                f"Fuzzy activation score ({activation_score:.2f} > 0.50) indicates high uncertainty "
                f"or query complexity requiring Layer 2 LLM."
            )

        return FuzzyRouterDecision(
            selected_layer=selected_layer,
            activation_score=activation_score,
            confidence=confidence,
            similarity=similarity,
            complexity=complexity,
            entity_density=entity_density,
            reason=reason,
            is_fallback=is_fallback,
            fired_rules=fired_rules,
        )


# Global singleton instance reused across requests
fuzzy_router = MamdaniFuzzyRouter()
