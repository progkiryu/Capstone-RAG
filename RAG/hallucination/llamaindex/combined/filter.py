# the verifier loads the same cross-encoder (with the same fallback), so share it
from .verifier import nli_model as _model

RELEVANCE_THRESHOLD = 0.5
FALLBACK_KEEP = 3


def _relevance(question, texts):
    pairs = [(question, text) for text in texts]
    if _model.num_labels == 1:
        return _model.predict(pairs)
    probabilities = _model.predict(pairs, apply_softmax=True)
    # a passage never entails a question (P(entailment) ~ 0 on every pair), so relevance
    # is the probability the passage is not judged contradictory / off-topic
    contradiction = _model.model.config.label2id["contradiction"]
    return 1 - probabilities[:, contradiction]


class NLIFilter:
    def filter(self, question, nodes):
        scores = _relevance(question, [node.get_content() for node in nodes])
        kept = [node for node, score in zip(nodes, scores) if score >= RELEVANCE_THRESHOLD]
        return kept or nodes[:FALLBACK_KEEP]
