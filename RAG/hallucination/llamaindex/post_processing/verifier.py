import re
import nltk
import numpy as np
from nltk.tokenize import PunktTokenizer
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
FALLBACK_MODEL = "cross-encoder/qnli-electra-base"
MIN_CLAIM_TOKENS = 3
CLAIM_SUPPORT_THRESHOLD = 0.5
PROVENANCE_THRESHOLD = 0.5


def _load_model():
    try:
        model, name = CrossEncoder(NLI_MODEL), NLI_MODEL
    except Exception as error:
        print(f"Verifier: could not load {NLI_MODEL} ({error})")
        model, name = CrossEncoder(FALLBACK_MODEL), FALLBACK_MODEL
    print(f"Verifier: loaded {name}")
    return model


nli_model = _load_model()
# the tokenizer sent_tokenize uses; spans let revision cut claims out of the draft's own text
_punkt = PunktTokenizer("english")


def answer_text(answer):
    if isinstance(answer, list):
        return " ".join(answer_text(item) for item in answer)
    for attribute in ("text", "content"):
        value = getattr(answer, attribute, None)
        if isinstance(value, str):
            return value
    return str(answer)


def split_claims(text):
    spans = []
    for start, end in _punkt.span_tokenize(text):
        if spans and len(nltk.word_tokenize(text[start:end])) < MIN_CLAIM_TOKENS:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def entailment(pairs):
    if nli_model.num_labels == 1:
        # QNLI scores (question, passage); the fallback model has no entailment class
        return nli_model.predict([(hypothesis, premise) for premise, hypothesis in pairs])
    probabilities = nli_model.predict(pairs, apply_softmax=True)
    return probabilities[:, nli_model.model.config.label2id["entailment"]]


def support_matrix(question, claims, chunks):
    # each claim against each chunk, not the concatenation: the cross-encoder truncates at
    # 512 tokens, so a ~16k-char context would be mostly unread. Whitespace is collapsed
    # because the PDFs extract one word per line.
    premises = [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks]

    def score(hypotheses):
        pairs = [(premise, hypothesis) for hypothesis in hypotheses for premise in premises]
        return entailment(pairs).reshape(len(hypotheses), len(premises))

    # drafts are often bare fragments ("1962") that entail nothing alone, while prefixing the
    # question hurts some full sentences; a claim counts as supported under either reading
    return np.maximum(score(claims), score([f"{question} {claim}" for claim in claims]))


def revise(draft, spans, keep):
    # rebuilt only from slices of the draft (claims plus the whitespace that preceded them),
    # so a revision can never contain text the draft did not
    pieces = []
    for i, ((start, end), kept) in enumerate(zip(spans, keep)):
        if kept:
            if pieces:
                pieces.append(draft[spans[i - 1][1]:start])
            pieces.append(draft[start:end])
    return "".join(pieces)


def provenance(claims, matrix, keep):
    records = [
        {"claim": claim, "chunk": int(row.argmax()), "nli": float(row.max())}
        for claim, row, kept in zip(claims, matrix, keep) if kept
    ]
    covered = sum(record["nli"] >= PROVENANCE_THRESHOLD for record in records)
    return records, (100 * covered / len(records) if records else 0.0)
