from dataclasses import dataclass, field
from .sanitiser import sanitise
from .detectors import heuristic_flag, classifier_score

REPLACEMENT = "[removed: possible injected instructions]"
FALLBACK = "I couldn't find reliable information to answer that."


@dataclass
class PreConfig:
    sanitise: bool = False
    heuristic: bool = False
    classifier: bool = False
    threshold: float = 0.9
    safe_prompt: bool = False
    spotlight: bool = False
    replace_instead_of_drop: bool = False


CONDITIONS = {
    "P0": PreConfig(),
    "P1": PreConfig(sanitise=True),
    "P2": PreConfig(sanitise=True, heuristic=True),
    "P3": PreConfig(sanitise=True, classifier=True),
    "P4": PreConfig(safe_prompt=True, spotlight=True),
    "P5": PreConfig(sanitise=True, heuristic=True, classifier=True, safe_prompt=True, spotlight=True),
}
DEFAULT_CONDITION = "P5"


@dataclass
class PreResult:
    chunks: list[str]
    log: list[dict] = field(default_factory=list)


def preprocess(chunks, cfg):
    out = PreResult(chunks=[])
    for i, raw in enumerate(chunks):
        text = sanitise(raw) if cfg.sanitise else raw
        hits = heuristic_flag(text) if cfg.heuristic else []
        score = classifier_score(text) if cfg.classifier else None
        flagged = bool(hits) or (score is not None and score >= cfg.threshold)
        out.log.append({"chunk": i, "heuristic": hits, "clf_score": score, "flagged": flagged})
        if not flagged:
            out.chunks.append(text)
        elif cfg.replace_instead_of_drop:
            out.chunks.append(REPLACEMENT)
    return out
