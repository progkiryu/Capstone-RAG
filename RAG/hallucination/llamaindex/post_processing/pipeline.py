from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .judge import Judge
from .consistency import Consistency, consistency_score, semantic_entropy
from .verifier import (
    CLAIM_SUPPORT_THRESHOLD, answer_text, provenance, revise, split_claims, support_matrix
)

NOT_FOUND = "Not found in context"


class LIPipeline:
    def __init__(self, llm, judge_llm, index_dir="./faiss_index"):
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.generator = Generator(llm)
        self.evaluator = Evaluator(judge_llm)
        self.judge = Judge(judge_llm)
        self.consistency = Consistency(llm, self.generator.prompt)
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
        # an abstaining draft has nothing to verify
        if not claims or draft.strip() == NOT_FOUND:
            return report

        matrix = support_matrix(question, claims, chunks)
        nli_support = [float(score) for score in matrix.max(axis=1)]
        judge_support = self.judge.judge(question, claims, chunks)
        if judge_support is None:
            support = nli_support
        else:
            support = [(nli + judged) / 2 for nli, judged in zip(nli_support, judge_support)]
        keep = [score >= CLAIM_SUPPORT_THRESHOLD for score in support]

        # sampled once and shared by the consistency score and the semantic entropy
        answers = [draft] + self.consistency.sample(question, chunks)
        entropy, clusters = semantic_entropy(question, answers)
        records, coverage = provenance(claims, matrix, keep)

        # every claim failing is left to the abstention gate
        if any(keep):
            report["answer"] = revise(draft, spans, keep)
        report.update(
            nli_support=nli_support, judge_support=judge_support, support=support,
            aggregate_support=sum(support) / len(support), kept=keep, dropped=keep.count(False),
            provenance=records, coverage=coverage, samples=answers[1:],
            consistency=consistency_score(answers), entropy=entropy, clusters=clusters,
        )
        return report

    def evaluate(self, question, reference):
        answer, _ = self.query(question)
        score, claim_scores = self.evaluator.evaluate(question, answer, reference)
        return {"answer": answer, "score": score, "claim_scores": claim_scores}
