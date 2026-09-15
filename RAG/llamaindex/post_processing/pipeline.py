from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever

class LIPipeline:
    def __init__(self, llm, judge_llm, index_dir="./faiss_index"):
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.generator = Generator(llm)
        self.evaluator = Evaluator(judge_llm)
        self.retriever = None
        # accumulate nodes so multiple ingest() calls index every file, not just the last
        self.nodes = None

    def ingest(self, file_path):
        nodes = self.indexer.load_and_split(file_path)
        if self.nodes is None:
            self.nodes = nodes
        else:
            self.nodes.extend(nodes)
        index = self.datastore.create(self.nodes)
        self.retriever = Retriever(index)

    def load_existing(self):
        index = self.datastore.load()
        self.retriever = Retriever(index)

    def query(self, question):
        nodes = self.retriever.retrieve(question)
        response = self.generator.generate(question, nodes)
        return response, nodes

    def evaluate(self, query, reference):
        nodes = self.retriever.retrieve(query)
        answer = self.generator.generate(query, nodes)

        claims = self.evaluator.split_into_claims(answer)

        claim_scores = self.evaluator.evaluate_claims(claims, query, reference)

        overall_score = self.evaluator.aggregate_claim_scores(claim_scores)

        return {
            "answer": answer,
            "claims": claims,
            "claim_scores": claim_scores,
            "score": overall_score
        }
