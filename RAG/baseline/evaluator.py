from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


class Evaluator:
    def __init__(self):
        # Lightweight embedding model
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def normalize(self, text: str) -> str:
        text = text.lower().strip()

        # remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # collapse whitespace
        text = re.sub(r"\s+", " ", text)

        return text

    def exact_match_score(self, prediction, reference):
        pred = self.normalize(prediction)
        ref = self.normalize(reference)

        return 100 if pred == ref else 0

    def fuzzy_match_score(self, prediction, reference):
        pred = self.normalize(prediction)
        ref = self.normalize(reference)

        return fuzz.token_sort_ratio(pred, ref)

    def embedding_score(self, prediction, reference):
        embeddings = self.embedding_model.encode(
            [prediction, reference]
        )

        sim = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return round(sim * 100, 2)

    def keyword_overlap_score(self, prediction, reference):
        pred_words = set(self.normalize(prediction).split())
        ref_words = set(self.normalize(reference).split())

        if not ref_words:
            return 0

        overlap = pred_words.intersection(ref_words)

        return round((len(overlap) / len(ref_words)) * 100, 2)

    def evaluate(self, prediction, reference):
        exact = self.exact_match_score(prediction, reference)
        fuzzy = self.fuzzy_match_score(prediction, reference)
        embedding = self.embedding_score(prediction, reference)
        keyword = self.keyword_overlap_score(prediction, reference)

        # weighted scoring
        final_score = (
            exact * 0.4 +
            fuzzy * 0.2 +
            embedding * 0.3 +
            keyword * 0.1
        )

        return {
            "final_score": round(final_score, 2),
            "exact_match": exact,
            "fuzzy_match": round(fuzzy, 2),
            "embedding_similarity": embedding,
            "keyword_overlap": keyword
        }