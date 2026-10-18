import nltk
# the consistency check loads the same MiniLM model, so share it
from .consistency import _embedder as _model

# 0.3 cut context ~42% on the Swan Lagoon set for a ~2-point drop in answer-word recall
SIMILARITY_THRESHOLD = 0.3


class Compressor:
    def compress(self, question, nodes):
        sentences = [nltk.sent_tokenize(node.get_content()) or [node.get_content()] for node in nodes]
        question_embedding = _model.encode(question, normalize_embeddings=True)
        similarities = _model.encode(
            [sentence for chunk in sentences for sentence in chunk], normalize_embeddings=True
        ) @ question_embedding

        compressed, offset = [], 0
        for node, chunk in zip(nodes, sentences):
            scores = similarities[offset:offset + len(chunk)]
            offset += len(chunk)
            best = int(scores.argmax())
            kept = [s for i, s in enumerate(chunk) if scores[i] >= SIMILARITY_THRESHOLD or i == best]
            # copy so the docstore's node keeps its full text
            compressed.append(node.model_copy(update={"text": " ".join(kept)}))
        return compressed
