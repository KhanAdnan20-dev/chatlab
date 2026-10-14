"""Priority Code Detector — detects code snippets, libraries, and code-related intent."""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeDetectionResult:
    is_code: bool
    code_text: str
    non_code_query: str
    detected_language: str = "python"
    detected_libraries: list[str] = field(default_factory=list)
    suggested_intent: Optional[str] = None
    needs_clarification: bool = False
    clarification_message: str = ""


class CodeDetector:
    """Analyzes raw input text for programming code structures prior to standard intent classification."""

    CODE_KEYWORDS = [
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\breturn\b",
        r"\bself\.\w+",
        r"\b__init__\b",
        r"\bnp\.\w+",
        r"\btorch\.\w+",
        r"\btf\.\w+",
        r"\bnn\.\w+",
        r"\bplt\.\w+",
        r"\bpd\.\w+",
    ]

    LIBRARY_MAP = {
        "numpy": [r"\bnumpy\b", r"\bnp\.\w+"],
        "pytorch": [r"\btorch\b", r"\bnn\.\w+"],
        "tensorflow": [r"\btensorflow\b", r"\btf\.\w+"],
        "scikit-learn": [r"\bsklearn\b"],
        "pandas": [r"\bpandas\b", r"\bpd\.\w+"],
    }

    def detect(self, text: str) -> CodeDetectionResult:
        if not text:
            return CodeDetectionResult(is_code=False, code_text="", non_code_query="")

        # Check markdown code fences
        fence_match = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
        code_inside_fence = ""
        if fence_match:
            code_inside_fence = fence_match.group(1).strip()
            text_without_fences = re.sub(r"```(?:\w+)?\n?(.*?)```", "", text, flags=re.DOTALL).strip()
        else:
            text_without_fences = text

        lines = [line for line in text.splitlines() if line.strip()]
        keyword_hits = 0
        for pat in self.CODE_KEYWORDS:
            if re.search(pat, text, re.IGNORECASE):
                keyword_hits += 1

        # Check structural code traits (indentation, colons, brackets, multiple lines)
        structural_traits = 0
        if any(line.startswith("    ") or line.startswith("\t") for line in lines):
            structural_traits += 1
        if any(line.strip().endswith(":") for line in lines):
            structural_traits += 1
        if text.count("=") >= 3 or text.count("(") >= 3:
            structural_traits += 1

        is_code = bool(code_inside_fence or keyword_hits >= 2 or (keyword_hits >= 1 and len(lines) >= 3) or (structural_traits >= 2 and len(lines) >= 5))

        if not is_code:
            return CodeDetectionResult(is_code=False, code_text="", non_code_query=text)

        code_text = code_inside_fence if code_inside_fence else text
        query_text = text_without_fences if fence_match else text

        # Detect libraries
        detected_libs = []
        for lib, patterns in self.LIBRARY_MAP.items():
            if any(re.search(pat, code_text, re.IGNORECASE) for pat in patterns):
                detected_libs.append(lib)

        # Classify intention relative to code
        q_lower = query_text.lower()
        has_explain_verb = any(kw in q_lower for kw in ["explain", "walkthrough", "understand", "what does", "how does", "describe"])
        has_debug_verb = any(kw in q_lower for kw in ["debug", "error", "bug", "fix", "wrong", "fail", "issue", "why is"])
        has_algorithm_verb = any(kw in q_lower for kw in ["algorithm", "what model", "which model", "identify"])

        suggested_intent = None
        needs_clarification = False
        clarification_msg = ""

        if has_debug_verb:
            suggested_intent = "problem_solving"
        elif has_explain_verb or has_algorithm_verb:
            suggested_intent = "code_explanation"
        elif len(query_text.strip()) == 0 or query_text.strip() == code_text.strip():
            # Code pasted with no prompt
            needs_clarification = True
            clarification_msg = (
                "I noticed you submitted a code snippet! Would you like me to "
                "explain how it works, debug an issue, or analyze the underlying algorithm?"
            )
            suggested_intent = "code_explanation"
        else:
            suggested_intent = "code_explanation"

        return CodeDetectionResult(
            is_code=True,
            code_text=code_text,
            non_code_query=query_text,
            detected_language="python",
            detected_libraries=detected_libs,
            suggested_intent=suggested_intent,
            needs_clarification=needs_clarification,
            clarification_message=clarification_msg,
        )


code_detector = CodeDetector()
