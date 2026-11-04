"""Frozen Base completion protocol v1; independent of the Instruct parser."""
from .ag_news import LABELS, TEMPLATE, build_prompt

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
BASE_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
POLICY = {
    "version": "base-completion-v1",
    "prompt": "Existing eight-shot text prompt, no chat template; ends exactly in Label:",
    "boundary": "First generated LF newline (even a leading newline), EOS, or 12-token cap",
    "parser": "First line before LF, strip whitespace, case-insensitive exact canonical label only",
    "reject": "Empty, role/Label prefixes, quotes, punctuation, explanations, ambiguous labels, token-cap termination",
    "invalid_scoring": "Null prediction, count as incorrect; no retries",
}


def build_base_prompt(demonstrations, article, max_chars=1000):
    if len(demonstrations) != 8 or any(sum(row['label'] == c for row in demonstrations) != 2 for c in range(4)):
        raise ValueError("Base smoke protocol requires eight demonstrations, two per class")
    return build_prompt(demonstrations, article, max_chars)


def completion_line(raw):
    return raw.split("\n", 1)[0]


def parse_base_label(completion):
    text = completion.strip()
    for index, label in enumerate(LABELS):
        if text.casefold() == label.casefold():
            return index
    raise ValueError(f"Invalid Base label under base-completion-v1: {completion!r}")
