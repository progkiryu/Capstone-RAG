import re
from functools import cache
from transformers import pipeline

CLASSIFIER_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
# leaves room for the [CLS] and [SEP] tokens within the model's 512-token limit
WINDOW_TOKENS = 510

PATTERNS = [re.compile(pattern, re.I | re.M) for pattern in [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |all )?(previous|above|system)",
    r"\byou are now\b",
    r"new instructions?\s*:",
    r"^\s*(system|assistant)\s*:",
    r"do not (tell|inform|mention).{0,30}\buser\b",
    r"(reveal|print|repeat) (your|the) (system )?prompt",
    r"!\[[^\]]*\]\(https?://",
]]


def heuristic_flag(text):
    return [pattern.pattern for pattern in PATTERNS if pattern.search(text)]


@cache
def _classifier():
    return pipeline("text-classification", model=CLASSIFIER_MODEL, truncation=True, max_length=512)


def classifier_score(text):
    classifier = _classifier()
    ids = classifier.tokenizer(text, add_special_tokens=False)["input_ids"]
    windows = [
        classifier.tokenizer.decode(ids[start:start + WINDOW_TOKENS])
        for start in range(0, len(ids), WINDOW_TOKENS)
    ] or [text]
    return max(
        result["score"] if result["label"] == "INJECTION" else 1 - result["score"]
        for result in classifier(windows)
    )
