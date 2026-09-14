from langchain_core.prompts import ChatPromptTemplate
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

class Evaluator:
    def __init__(self, llm):
        # judge model is selected via --judge in main.py
        self.llm = llm

        # Prompt template for evaluation
        self.prompt = ChatPromptTemplate.from_template("""
        You are an evaluation system.

        Question: {question}
        Ground Truth Answer: {reference}
        Model Answer: {prediction}

        Evaluate if the model answer is correct.

        Respond only with a number from 0 to 100, 100 meaning completely correct,
        0 meaning completely incorrect.
        """)

        
        self.chain = self.prompt | self.llm

    def evaluate(self, prediction, reference, question):
        response = self.chain.invoke({
            "question": question,
            "reference": reference,
            "prediction": prediction
        })

        return response.content

    def split_into_claims(self, text):
        """
        Split model output into individual claims using sentence tokenization.
        Handles both strings and AIMessage objects from LangChain.
        """
        # If text is an AIMessage, extract content
        if hasattr(text, "content"):
            text = text.content
        elif isinstance(text, list):
            # If a list of messages, join their content
            text = " ".join([msg.content if hasattr(msg, "content") else str(msg) for msg in text])

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