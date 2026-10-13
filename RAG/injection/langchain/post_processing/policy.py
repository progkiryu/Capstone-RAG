from dataclasses import dataclass
from .checks import (
    canary_leaked, exfil_findings, off_task_sentences, split_sentences, strip_findings,
    ungrounded_sentences,
)
from .judge import judge

FALLBACK = "I couldn't produce a reliable answer from the available documents."
ALLOWED_DOMAINS = set()


@dataclass
class PostConfig:
    canary: bool = False
    exfil: bool = False
    alignment: bool = False
    align_threshold: float = 0.2
    grounded: bool = False
    grounded_chunks: str = "all"
    judge: bool = False
    regenerate: bool = False
    min_chars: int = 20


Q1 = dict(canary=True, exfil=True)
CONDITIONS = {
    "Q0": PostConfig(),
    "Q1": PostConfig(**Q1),
    "Q2": PostConfig(**Q1, alignment=True),
    "Q3a": PostConfig(**Q1, grounded=True, grounded_chunks="all"),
    "Q3b": PostConfig(**Q1, grounded=True, grounded_chunks="unflagged"),
    "Q4": PostConfig(**Q1, judge=True),
    "Q5": PostConfig(**Q1, alignment=True, grounded=True, grounded_chunks="unflagged", judge=True),
    "Q6": PostConfig(**Q1, alignment=True, grounded=True, grounded_chunks="unflagged", judge=True,
                     regenerate=True),
}
DEFAULT_CONDITION = "Q5"


def postprocess(question, response, chunks, canary, allowed, cfg, call_judge=None, regen=None):
    log = {"actions": []}

    if cfg.canary and canary_leaked(response, canary):
        return FALLBACK, {**log, "actions": ["block:canary"]}

    flagged_text = []
    if cfg.exfil:
        found = exfil_findings(response, question, allowed)
        if found:
            response = strip_findings(response, found)
            log["actions"].append(f"strip:{len(found)}")
            flagged_text += found

    bad = set()
    if cfg.alignment:
        bad |= set(off_task_sentences(question, response, cfg.align_threshold))
    if cfg.grounded:
        bad |= set(ungrounded_sentences(response, chunks))
    if bad:
        response = " ".join(sentence for sentence in split_sentences(response) if sentence not in bad)
        log["actions"].append(f"redact:{len(bad)}")
        flagged_text += sorted(bad)

    suspicious = False
    if cfg.judge and call_judge:
        verdict = judge(question, response, call_judge)
        log["judge"] = verdict
        suspicious = verdict.get("verdict") != "clean"

    # only an answer cut down by stripping or redaction counts as too short, so short
    # untouched answers ("1962") are not blocked when no check fired
    too_short = bool(flagged_text) and len(response) < cfg.min_chars
    if suspicious or too_short:
        if cfg.regenerate and regen:
            log["actions"].append("regenerate")
            return regen(" ".join(flagged_text) or response), log
        return FALLBACK, {**log, "actions": log["actions"] + ["block"]}

    return response, log
