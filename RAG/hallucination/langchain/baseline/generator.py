from langchain_core.prompts import ChatPromptTemplate


class Generator:
    def __init__(self, llm):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template("""
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
        """)

    def query(self, query, docs):
        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        chain = self.prompt | self.llm
        return chain.invoke({
            "context": context,
            "question": query
        })
