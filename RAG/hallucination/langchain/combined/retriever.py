import re
from langchain_community.retrievers import BM25Retriever

K = 16
MMR_FETCH_K = 48
MMR_LAMBDA = 0.5
DENSE_K = 10
# FAISS IndexFlatL2 returns squared L2; for unit-length text-embedding-3-small vectors
# 0.35 corresponds to cosine similarity >= 0.825
DENSE_MAX_DISTANCE = 0.35
QUERY_MATCH_BONUS = 2

SIGNALS = [
    (re.compile(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"), 5),           # ABN
    (re.compile(r"(\+61\s?\d{1,4}\s?\d{3}\s?\d{3}|\b0\d{8,9}\b)"), 4),  # AU phone
    (re.compile(r"\+?\d[\d\s\-()]{7,}\d"), 3),                        # international phone
    (re.compile(r"\b\d{6,}\b"), 2),                                   # long identifier
]


def normalize_query(query):
    return re.sub(r"\s+", " ", query.lower().strip())


def signal_score(text):
    return sum(weight for pattern, weight in SIGNALS if pattern.search(text))


class Retriever:
    def __init__(self, vectorstore, docs):
        self.vectorstore = vectorstore
        self.bm25 = BM25Retriever.from_documents(docs, k=K)
        self.last_counts = {}

    def retrieve(self, query):
        query = normalize_query(query)
        embedding = self.vectorstore.embeddings.embed_query(query)

        mmr_docs = self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding, k=K, fetch_k=MMR_FETCH_K, lambda_mult=MMR_LAMBDA
        )

        dense_docs = [
            doc for doc, distance in
            self.vectorstore.similarity_search_with_score_by_vector(embedding, k=DENSE_K)
            if distance <= DENSE_MAX_DISTANCE
        ]

        bm25_docs = self.bm25.invoke(query)

        unique_docs, seen = [], set()
        for doc in mmr_docs + dense_docs + bm25_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        def rank_score(doc):
            text = doc.page_content
            bonus = QUERY_MATCH_BONUS if query in normalize_query(text) else 0
            return signal_score(text) + bonus

        # stable sort: ties keep MMR, dense, BM25 pool order
        unique_docs.sort(key=rank_score, reverse=True)

        self.last_counts = {
            "mmr": len(mmr_docs), "dense": len(dense_docs),
            "bm25": len(bm25_docs), "pooled": len(unique_docs),
        }
        return unique_docs[:K]
