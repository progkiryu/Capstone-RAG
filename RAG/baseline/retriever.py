class Retriever:
    def __init__(self, vectorstore, k=16):
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": int(k)}
        )

    def retrieve(self, query):
        return self.retriever.invoke(query)