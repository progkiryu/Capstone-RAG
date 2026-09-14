from llama_index.core.prompts import ChatPromptTemplate

class Generator:
    def __init__(self, llm):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([("user", """
            Answer the question using ONLY the context below.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """)])

    def generate(self, query, nodes):
        context = "\n\n".join([node.get_content() for node in nodes])

        return self.llm.predict(
            self.prompt,
            context=context,
            question=query
        )
