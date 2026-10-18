import re
from llama_index.core.prompts import ChatPromptTemplate
from nltk.tokenize import sent_tokenize


# measurement harness only: the one module that reads the ground truth. The defences
# (verifier, judge, consistency) must never import it or receive `reference`.
class Evaluator:
    def __init__(self, judge_llm):
        self.llm = judge_llm
        self.prompt = ChatPromptTemplate.from_messages([("user", """
        You are an evaluation system.

        Question: {question}
        Ground Truth Answer: {reference}
        Model Answer: {prediction}

        Evaluate if the model answer is correct.

        Respond only with a number from 0 to 100, 100 meaning completely correct,
        0 meaning completely incorrect.
        """)])

    def score_claim(self, question, claim, reference):
        text = self.llm.predict(
            self.prompt,
            question=question,
            reference=reference,
            prediction=claim
        )
        # a reply with no number scores 0 instead of crashing
        match = re.search(r"\d+(?:\.\d+)?", text)
        return float(match.group()) if match else 0.0

    def evaluate(self, question, answer, reference):
        claims = sent_tokenize(answer) or [answer]
        claim_scores = {claim: self.score_claim(question, claim, reference) for claim in claims}
        return sum(claim_scores.values()) / len(claim_scores), claim_scores
