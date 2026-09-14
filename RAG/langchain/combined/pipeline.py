from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever


class LCPipeline:

    def __init__(self, llm, judge_llm, index_dir="./faiss_index"):

        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)

        self.generator = Generator(llm)
        self.evaluator = Evaluator(judge_llm)

        self.retriever = None

        # store docs for BM25
        self.docs = None

    def ingest(self, file_path):

        docs = self.indexer.load_and_split(file_path)

        # store docs for BM25 retriever
        if self.docs is None:
            self.docs = docs
        else:
            self.docs.extend(docs)

        # create / update vector store
        db = self.datastore.create(self.docs)

        self.retriever = Retriever(db, self.docs)
    def load_existing(self):
        db = self.datastore.load()
        if self.docs is None:
            raise ValueError(
                "BM25 retriever requires documents. "
                "Load or rebuild docs before using load_existing()."
            )

        self.retriever = Retriever(db, self.docs)
    def query(self, question):

        docs = self.retriever.retrieve(question)

        response = self.generator.query(question, docs)

        return response

    def evaluate(self, query, reference):
        docs = self.retriever.retrieve(query)
        answer = self.generator.query(query, docs)

        claims = self.evaluator.split_into_claims(answer)

        claim_scores = self.evaluator.evaluate_claims(claims, query, reference)

        overall_score = self.evaluator.aggregate_claim_scores(claim_scores)

        return {
            "answer": answer,
            "claims": claims,
            "claim_scores": claim_scores,
            "score": overall_score
        }