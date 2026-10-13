import nltk
from langchain_core.documents import Document
# the consistency check loads the same MiniLM model, so share it
from .consistency import _embedder as _model

# 0.3 cut context ~42% on the Swan Lagoon set for a ~2-point drop in answer-word recall
SIMILARITY_THRESHOLD = 0.3


class Compressor:
    def compress(self, question, docs):
        sentences = [nltk.sent_tokenize(doc.page_content) or [doc.page_content] for doc in docs]
        question_embedding = _model.encode(question, normalize_embeddings=True)
        similarities = _model.encode(
            [sentence for chunk in sentences for sentence in chunk], normalize_embeddings=True
        ) @ question_embedding

        compressed, offset = [], 0
        for doc, chunk in zip(docs, sentences):
            scores = similarities[offset:offset + len(chunk)]
            offset += len(chunk)
            best = int(scores.argmax())
            kept = [s for i, s in enumerate(chunk) if scores[i] >= SIMILARITY_THRESHOLD or i == best]
            compressed.append(Document(page_content=" ".join(kept), metadata=dict(doc.metadata)))
        return compressed
