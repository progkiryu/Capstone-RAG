from llama_index.core.prompts import ChatPromptTemplate
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

class Evaluator:
    def __init__(self, llm):
        # judge model is selected via --judge in main.py
        self.llm = llm

        # Prompt template for evaluation
        self.prompt = ChatPromptTemplate.from_messages([("user", """
        You are an evaluation system.

        Question: {question}
        Ground Truth Answer: {reference}
        Model Answer: {prediction}

        Evaluate if the model answer is correct.

        Respond only with a number from 0 to 100, 100 meaning completely correct,
        0 meaning completely incorrect.
        """)])

    def evaluate(self, prediction, reference, question):
        return self.llm.predict(
            self.prompt,
            question=question,
            reference=reference,
            prediction=prediction
        )

    def split_into_claims(self, text):
        """
        Split model output into individual claims using sentence tokenization.
        LlamaIndex's llm.predict() already returns a plain string.
        """
        # Use NLTK to split into sentences
        return sent_tokenize(str(text))

    def evaluate_claims(self, claims, question, reference):
        """
        Evaluate each claim individually using retrieved context.
        """
        claim_scores = {}
        for claim in claims:
            # Combine retrieved documents into a single context string
            score = self.evaluate(prediction=claim, reference=reference, question=question)
            claim_scores[claim] = score
        return claim_scores

    def aggregate_claim_scores(self, claim_scores):
        """
        Aggregate individual claim scores into an overall score.
        For simplicity, we can take the average of claim scores.
        """
        total_score = sum(float(score) for score in claim_scores.values())
        average_score = total_score / len(claim_scores) if claim_scores else 0
        return average_score
