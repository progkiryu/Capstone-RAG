class Retriever:
    def __init__(self, index, k=16):
        self.retriever = index.as_retriever(
            similarity_top_k=int(k)
        )

    def retrieve(self, query):
        return self.retriever.retrieve(query)
