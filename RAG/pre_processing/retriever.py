import re
from langchain_community.retrievers import BM25Retriever


class Retriever:
    def __init__(self, vectorstore, docs):

        self.vectorstore = vectorstore
        self.k = 16

        # FAISS distance threshold
        # lower = better match
        self.score_threshold = 0.35

        # BM25 lexical retriever
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = self.k

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

    def retrieve(self, query):

        query = self._normalize_query(query)

        # MMR retrieval
        mmr_docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.k,
            fetch_k=12,
            lambda_mult=0.5
        )

        # Semantic retrieval (FAISS)
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=10
        )

        semantic_docs = []

        for doc, score in docs_and_scores:
            # FAISS score: lower is better
            if score <= self.score_threshold:
                semantic_docs.append(doc)

        # BM25 retrieval
        bm25_docs = self.bm25_retriever.invoke(query)

        # Boost documents containing numeric signals
        boosted_docs = []

        for doc in bm25_docs + semantic_docs:

            boost = self._extract_numeric_signals(doc.page_content)

            if boost > 0:
                boosted_docs.append(doc)

        # combine and deduplicate
        combined = (
            boosted_docs
            + bm25_docs
            + mmr_docs
            + semantic_docs
        )

        unique_docs = []
        seen = set()

        for doc in combined:
            content_hash = hash(doc.page_content)

            if content_hash not in seen:
                unique_docs.append(doc)
                seen.add(content_hash)

        # ranking function
        def score_doc(doc):

            text = doc.page_content.lower()
            score = 0

            # query relevance
            if query in text:
                score += 2

            # numeric signal importance
            score += self._extract_numeric_signals(text)

            return score

        unique_docs.sort(
            key=score_doc,
            reverse=True
        )

        return unique_docs[:self.k]