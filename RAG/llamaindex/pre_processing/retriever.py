import re
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.retrievers.bm25 import BM25Retriever


class Retriever:
    def __init__(self, index, nodes, embed_model):

        self.index = index
        self.embed_model = embed_model
        self.k = 16

        # FAISS distance threshold
        # lower = better match
        self.score_threshold = 0.35

        # BM25 lexical retriever
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self.k
        )

    # query normalisation (pre-processing)
    def _normalize_query(self, query):
        query = query.lower().strip()
        query = re.sub(r"\s+", " ", query)
        return query

    def _contains_numeric_pattern(self, query):
        return bool(re.search(r"\d{4,}", query))

    def _extract_numeric_signals(self, text):
        score = 0

        # ABN (Australian Business Number)
        if re.search(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b", text):
            score += 5

        # Australian phone numbers
        if re.search(r"(\+61\s?\d{1,4}\s?\d{3}\s?\d{3}|\b0\d{8,9}\b)", text):
            score += 4

        # Generic international phone numbers
        if re.search(r"\+?\d[\d\s\-()]{7,}\d", text):
            score += 3

        # Long numeric identifiers (invoice, account numbers, etc.)
        if re.search(r"\b\d{6,}\b", text):
            score += 2

        return score

    def _max_marginal_relevance(self, query, k, fetch_k, lambda_mult):
        # FaissVectorStore has no MMR mode, so re-rank its fetch_k nearest
        # vectors with LlamaIndex's MMR, as LangChain's FAISS store does
        vector_store = self.index.vector_store
        query_embedding = self.embed_model.get_query_embedding(query)

        result = vector_store.query(VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=fetch_k
        ))

        embeddings = [
            vector_store.client.reconstruct(int(vector_id)).tolist()
            for vector_id in result.ids
        ]

        _, mmr_ids = get_top_k_mmr_embeddings(
            query_embedding,
            embeddings,
            similarity_top_k=k,
            embedding_ids=result.ids,
            mmr_threshold=lambda_mult
        )

        node_ids = [self.index.index_struct.nodes_dict[vector_id] for vector_id in mmr_ids]
        return self.index.docstore.get_nodes(node_ids)

    def retrieve(self, query):

        query = self._normalize_query(query)

        # MMR retrieval
        mmr_nodes = self._max_marginal_relevance(
            query,
            k=self.k,
            fetch_k=12,
            lambda_mult=0.5
        )

        # Semantic retrieval (FAISS)
        nodes_and_scores = self.index.as_retriever(
            similarity_top_k=10
        ).retrieve(query)

        semantic_nodes = []

        for result in nodes_and_scores:
            # FAISS score: lower is better
            if result.score <= self.score_threshold:
                semantic_nodes.append(result.node)

        # BM25 retrieval
        bm25_nodes = [
            result.node for result in self.bm25_retriever.retrieve(query)
        ]

        # Boost nodes containing numeric signals
        boosted_nodes = []

        for node in bm25_nodes + semantic_nodes:

            boost = self._extract_numeric_signals(node.get_content())

            if boost > 0:
                boosted_nodes.append(node)

        # combine and deduplicate
        combined = (
            boosted_nodes
            + bm25_nodes
            + mmr_nodes
            + semantic_nodes
        )

        unique_nodes = []
        seen = set()

        for node in combined:
            content_hash = hash(node.get_content())

            if content_hash not in seen:
                unique_nodes.append(node)
                seen.add(content_hash)

        # ranking function
        def score_node(node):

            text = node.get_content().lower()
            score = 0

            # query relevance
            if query in text:
                score += 2

            # numeric signal importance
            score += self._extract_numeric_signals(text)

            return score

        unique_nodes.sort(
            key=score_node,
            reverse=True
        )

        return unique_nodes[:self.k]
