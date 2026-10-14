"""Context-Aware Follow-Up Resolver — resolves ambiguous multi-turn user queries."""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResolvedQuery:
    original_message: str
    resolved_message: str
    has_context: bool
    requires_clarification: bool = False
    clarification_message: str = ""


class ContextResolver:
    """Resolves short or ambiguous follow-up user queries using preceding conversation turns."""

    AMBIGUOUS_PATTERNS = [
        r"^explain(?:\s+(?:this|that|it|more))?\s*$",
        r"^explain\s+with\s+an?\s+example\s*$",
        r"^how\s+does\s+it\s+work\s*$",
        r"^tell\s+me\s+more\s*$",
        r"^more\s+details?\s*$",
        r"^why\s*$",
        r"^why\s+(?:is\s+that|so)\s*$",
        r"^how\s+can\s+i\s+(?:reduce|fix|improve)\s+it\s*$",
        r"^which\s+one\s+is\s+better\s*$",
        r"^give\s+an?\s+example\s*$",
    ]

    def resolve(self, message: str, history: list[dict]) -> ResolvedQuery:
        clean_msg = message.strip()
        normalized_lower = re.sub(r"[^\w\s]", "", clean_msg.lower()).strip()

        # Check if query matches a bare ambiguous follow-up pattern
        is_ambiguous_pattern = any(re.search(pat, normalized_lower) for pat in self.AMBIGUOUS_PATTERNS)
        is_short_ambiguous = normalized_lower in ["explain", "why", "tell me more", "details", "more"]

        if not (is_ambiguous_pattern or is_short_ambiguous):
            return ResolvedQuery(
                original_message=clean_msg,
                resolved_message=clean_msg,
                has_context=False,
            )

        # Look backward through history for the last user topic
        last_user_query = ""
        for turn in reversed(history):
            if turn.get("role") == "user":
                prev_text = turn.get("content", "").strip()
                # Ignore previous meta commands or greetings
                if prev_text and not any(meta in prev_text.lower() for meta in ["hello", "hi", "clear", "start quiz", "bye"]):
                    last_user_query = prev_text
                    break

        if not last_user_query:
            # Standalone ambiguous query without context
            if is_ambiguous_pattern or clean_msg in ["explain", "why", "tell me more"]:
                return ResolvedQuery(
                    original_message=clean_msg,
                    resolved_message=clean_msg,
                    has_context=False,
                    requires_clarification=True,
                    clarification_message="Sure! What AI/ML concept, algorithm, or code snippet would you like me to explain?",
                )
            return ResolvedQuery(
                original_message=clean_msg,
                resolved_message=clean_msg,
                has_context=False,
            )

        # Context exists! Combine user's follow-up directive with last user topic
        resolved_text = f"{clean_msg} regarding '{last_user_query}'"

        if normalized_lower in ["explain", "explain this"]:
            resolved_text = f"Explain '{last_user_query}' in detail"
        elif normalized_lower in ["why", "why so", "why is that"]:
            resolved_text = f"Explain why '{last_user_query}' occurs in practice"
        elif "example" in normalized_lower:
            resolved_text = f"Explain '{last_user_query}' with a concrete example"
        elif "reduce" in normalized_lower or "fix" in normalized_lower:
            resolved_text = f"How can I reduce or fix '{last_user_query}'?"

        return ResolvedQuery(
            original_message=clean_msg,
            resolved_message=resolved_text,
            has_context=True,
        )


context_resolver = ContextResolver()
