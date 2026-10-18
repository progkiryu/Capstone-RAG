import re
from llama_index.core.prompts import ChatPromptTemplate

CORRECT, AMBIGUOUS, INCORRECT = "CORRECT", "AMBIGUOUS", "INCORRECT"
NOT_FOUND = "Not found in context"
# above the largest retrieved context on the Swan Lagoon set (~11k chars), so truncation
# cannot hide an answer and cause a false INCORRECT
GRADE_CONTEXT_CHARS = 12000

GRADE_PROMPT = """
You are grading whether retrieved passages can answer a question.

Reply with exactly one word:
- CORRECT if the passages contain the information needed to answer the question
- AMBIGUOUS if the passages are related but only partly answer it, or it is unclear
- INCORRECT if the passages do not contain the information needed

Question:
{question}

Passages:
{context}

Grade:
"""


def parse_grade(text):
    found = set(re.findall(r"\b(CORRECT|AMBIGUOUS|INCORRECT)\b", text.strip().upper()))
    if len(found) == 1:
        return found.pop(), True
    # fail open: an unreadable grade must never suppress an answer
    return CORRECT, False


class Grader:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([("user", GRADE_PROMPT)])

    def grade(self, question, nodes):
        context = "\n\n".join(node.get_content() for node in nodes)[:GRADE_CONTEXT_CHARS]
        return parse_grade(self.llm.predict(self.prompt, question=question, context=context))
