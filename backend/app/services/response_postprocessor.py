"""Response Postprocessor — cleans, normalises, repairs formatting, and ensures complete responses."""
import logging
import re

logger = logging.getLogger(__name__)
MAX_RESPONSE_CHARS = 2500

SPECIAL_TOKENS = [
    r"<start_of_turn>model\s*",
    r"<start_of_turn>user\s*",
    r"<start_of_turn>",
    r"<end_of_turn>",
    r"<bos>",
    r"<eos>",
    r"<pad>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
]


def fix_unicode_artifacts(text: str) -> str:
    """Repairs encoding glitches, replacement characters, and typographic artifacts."""
    if not text:
        return ""
    # Fix unicode replacement character \ufffd (commonly from mangled em-dash or smart quotes)
    cleaned = text.replace("\ufffd", " — ")
    # Normalize excessive dashes
    cleaned = re.sub(r" —\s*— ", " — ", cleaned)
    # Normalize common broken smart quote encodings
    cleaned = cleaned.replace("â€œ", '"').replace("â€", '"').replace("â€™", "'").replace("â€”", " — ")
    return cleaned


def repair_markdown_syntax(text: str) -> str:
    """Balances unclosed code blocks, math fences, and inline formatting tags."""
    if not text:
        return ""
    cleaned = text

    # Balance code fences (```)
    code_fence_count = len(re.findall(r"```", cleaned))
    if code_fence_count % 2 != 0:
        cleaned = cleaned.rstrip() + "\n```"

    # Balance block math ($$)
    math_block_count = len(re.findall(r"\$\$", cleaned))
    if math_block_count % 2 != 0:
        cleaned = cleaned.rstrip() + "$$"

    # Balance bold tags (**) outside of code blocks
    # (Exclude lines inside code blocks for bold count)
    non_code_parts = re.split(r"```.*?```", cleaned, flags=re.DOTALL)
    total_bold = sum(part.count("**") for part in non_code_parts)
    if total_bold % 2 != 0:
        cleaned = cleaned.rstrip() + "**"

    return cleaned


def repair_sentence_boundaries(text: str) -> str:
    """
    Ensures the response does not terminate abruptly mid-sentence or mid-clause.
    If truncated at token budget, cleanly concludes at the last coherent sentence or closes open parens.
    """
    if not text or len(text.strip()) < 40:
        return text

    stripped = text.strip()

    # Terminal punctuation that indicates clean sentence endings
    clean_endings = ('.', '!', '?', '```', '$$', '---', ':')
    if stripped.endswith(clean_endings):
        return stripped

    # Check for unclosed open parenthesis in the final line
    last_line = stripped.splitlines()[-1] if stripped.splitlines() else stripped
    open_paren = last_line.count("(")
    close_paren = last_line.count(")")
    if open_paren > close_paren:
        # If the unclosed clause is short, complete it with parenthesis and period
        stripped = stripped.rstrip() + ")."
        return stripped

    # If the response ends on an incomplete clause, look for the last complete sentence
    last_period = max(stripped.rfind('. '), stripped.rfind('.\n'), stripped.rfind('! '), stripped.rfind('? '))
    if last_period > len(stripped) * 0.55:
        # Trim back to the clean sentence boundary if we retain at least 55% of the content
        return stripped[:last_period + 1].strip()

    # Otherwise, cleanly terminate with a period
    if not stripped.endswith(('.', '!', '?')):
        stripped = stripped.rstrip(" ,;:-") + "."

    return stripped


def postprocess(text: str, max_chars: int = MAX_RESPONSE_CHARS) -> str:
    """Apply postprocessing filters, formatting normalizers, and token sanitization to raw response text."""
    from backend.app.config import settings

    fallback = settings.FALLBACK_RESPONSE

    if not text or not text.strip():
        logger.warning("Postprocessor received empty text; returning fallback.")
        return fallback

    cleaned = text

    # Strip special tokens
    for token_pat in SPECIAL_TOKENS:
        cleaned = re.sub(token_pat, "", cleaned, flags=re.IGNORECASE)

    # Strip prefix labels like 'model\n', 'assistant:\n', 'ChatLab:\n'
    cleaned = re.sub(r"^(?:model|assistant|chatlab)\s*:\s*", "", cleaned, flags=re.IGNORECASE)

    # Strip control chars (keep newlines and tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # Fix unicode artifacts
    cleaned = fix_unicode_artifacts(cleaned)

    # Collapse 3+ consecutive newlines to 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    # Repair truncated sentences if needed
    cleaned = repair_sentence_boundaries(cleaned)

    # Repair unclosed markdown fences (code blocks, bold, math)
    cleaned = repair_markdown_syntax(cleaned)

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + " ..."
        logger.debug("Response truncated to %d chars.", max_chars)

    return cleaned or fallback
