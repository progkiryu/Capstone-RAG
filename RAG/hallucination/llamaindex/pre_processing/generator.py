from llama_index.core.prompts import ChatPromptTemplate


class Generator:
    def __init__(self, llm):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([("user", """
        You are a retrieval-grounded QA system.
        Answer using ONLY the context below.

        Rules:
        - Do not explain
        - Do not guess
        - If answer is not in context, say "Not found in context"

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)])

    def query(self, query, nodes):
        context = "\n\n".join([
            node.get_content() for node in nodes
        ])

        return self.llm.predict(
            self.prompt,
            context=context,
            question=query
        )
