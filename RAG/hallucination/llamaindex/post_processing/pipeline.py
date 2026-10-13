from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .verifier import (
    CLAIM_SUPPORT_THRESHOLD, answer_text, provenance, revise, split_claims, support_matrix
)

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
        draft = answer_text(self.generator.query(question, nodes))
        self.last_report = self.verify(question, draft, [node.get_content() for node in nodes])
        return self.last_report["answer"], nodes

    # sees only the question, the draft and the retrieved context, never the reference
    def verify(self, question, draft, chunks):
        spans = split_claims(draft)
        claims = [draft[start:end] for start, end in spans]
        report = {"draft": draft, "claims": claims, "answer": draft}
        if not claims:
            return report

        matrix = support_matrix(question, claims, chunks)
        support = [float(score) for score in matrix.max(axis=1)]
        keep = [score >= CLAIM_SUPPORT_THRESHOLD for score in support]
        records, coverage = provenance(claims, matrix, keep)

        # every claim failing is left to the abstention gate
        if any(keep):
            report["answer"] = revise(draft, spans, keep)
        report.update(
            nli_support=support, kept=keep, dropped=keep.count(False),
            provenance=records, coverage=coverage,
        )
        return report

    def evaluate(self, question, reference):
        answer, _ = self.query(question)
        score, claim_scores = self.evaluator.evaluate(question, answer, reference)
        return {"answer": answer, "score": score, "claim_scores": claim_scores}
