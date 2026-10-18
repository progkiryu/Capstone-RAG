import re
from llama_index.core.prompts import ChatPromptTemplate

MAX_LENGTH_RATIO = 3

RULES = """
Rules:
- Keep every proper noun (people, places, businesses) exactly as written
- Keep every number and identifier (phone numbers, ABNs, years, codes) exactly as written
- Use plain words only: no AND/OR operators, brackets or alternative phrasings
- Output only the search query: no explanation, no quotes, no labels
"""

REWRITE_PROMPT = """
Rewrite the question below as a single search query for retrieving passages from a document collection.
""" + RULES + """
Question:
{question}

Search query:
"""

RETRY_PROMPT = """
Rewrite the question below as a single search query for retrieving passages from a document collection.
The previous search query, "{previous}", returned passages that only partly matched. Write a different query.
""" + RULES + """
Question:
{question}

Search query:
"""


def accept(original, rewrite):
    if not rewrite or len(rewrite) > MAX_LENGTH_RATIO * len(original):
        return False
    rewrite_numbers = set(re.findall(r"\d+", rewrite))
    return all(number in rewrite_numbers for number in re.findall(r"\d+", original))


class Rewriter:
    def __init__(self, llm):
        self.llm = llm
        self.rewrite_prompt = ChatPromptTemplate.from_messages([("user", REWRITE_PROMPT)])
        self.retry_prompt = ChatPromptTemplate.from_messages([("user", RETRY_PROMPT)])

    def rewrite(self, question, previous=None):
        if previous is None:
            text = self.llm.predict(self.rewrite_prompt, question=question)
        else:
            text = self.llm.predict(self.retry_prompt, question=question, previous=previous)
        lines = text.strip().splitlines()
        rewrite = lines[0].strip().strip("\"'") if lines else ""
        if accept(question, rewrite):
            return rewrite, True
        return question, False
