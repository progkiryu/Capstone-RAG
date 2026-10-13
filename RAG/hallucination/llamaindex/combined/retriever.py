import re
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.retrievers.bm25 import BM25Retriever

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
    def __init__(self, index, nodes, embed_model):
        self.index = index
        self.vector_store = index.vector_store
        self.embed_model = embed_model
        self.bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=K)
        self.last_counts = {}

    def _nodes(self, vector_ids):
        node_ids = [self.index.index_struct.nodes_dict[vector_id] for vector_id in vector_ids]
        return self.index.docstore.get_nodes(node_ids)

    def retrieve(self, query):
        query = normalize_query(query)
        embedding = self.embed_model.get_query_embedding(query)

        fetched = self.vector_store.query(VectorStoreQuery(
            query_embedding=embedding, similarity_top_k=MMR_FETCH_K
        ))

        # FaissVectorStore has no MMR mode, so re-rank the fetch_k nearest with
        # LlamaIndex's MMR, as LangChain's FAISS store does
        candidates = [
            self.vector_store.client.reconstruct(int(vector_id)).tolist()
            for vector_id in fetched.ids
        ]
        _, mmr_ids = get_top_k_mmr_embeddings(
            embedding, candidates,
            similarity_top_k=K, embedding_ids=fetched.ids, mmr_threshold=MMR_LAMBDA
        )
        mmr_nodes = self._nodes(mmr_ids)

        # IndexFlatL2 search is exact, so the dense top-k is the head of the fetch;
        # FaissVectorStore reports raw L2 distances in "similarities"
        dense_nodes = self._nodes([
            vector_id for vector_id, distance in
            zip(fetched.ids[:DENSE_K], fetched.similarities[:DENSE_K])
            if distance <= DENSE_MAX_DISTANCE
        ])

        bm25_nodes = [result.node for result in self.bm25.retrieve(query)]

        unique_nodes, seen = [], set()
        for node in mmr_nodes + dense_nodes + bm25_nodes:
            content_hash = hash(node.get_content())
            if content_hash not in seen:
                seen.add(content_hash)
                unique_nodes.append(node)

        def rank_score(node):
            text = node.get_content()
            bonus = QUERY_MATCH_BONUS if query in normalize_query(text) else 0
            return signal_score(text) + bonus

        # stable sort: ties keep MMR, dense, BM25 pool order
        unique_nodes.sort(key=rank_score, reverse=True)

        self.last_counts = {
            "mmr": len(mmr_nodes), "dense": len(dense_nodes),
            "bm25": len(bm25_nodes), "pooled": len(unique_nodes),
        }
        return unique_nodes[:K]
