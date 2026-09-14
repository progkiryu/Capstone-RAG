from langchain_core.prompts import ChatPromptTemplate

class Generator:
    def __init__(self, llm):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template("""
            Answer the question using ONLY the context below.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """)

    def generate(self, query, docs):
        context = "\n\n".join([doc.page_content for doc in docs])

        chain = self.prompt | self.llm
        return chain.invoke({
            "context": context,
            "question": query
        })
