import time
from contextlib import contextmanager

from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .judge import Judge
from .consistency import SAMPLE_COUNT, Consistency, consistency_score, semantic_entropy
from .verifier import (
    CLAIM_SUPPORT_THRESHOLD, answer_text, provenance, revise, split_claims, support_matrix
)

NOT_FOUND = "Not found in context"
# fixed from the observed support/entropy distributions (no reference answers consulted):
# abstain when claims are on average unsupported, or when the sampled answers are spread
# over more meanings than an even three-way split (ln 3)
SUPPORT_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 1.10


@contextmanager
def timed(timings, stage):
    start = time.perf_counter()
    yield
    timings[stage] = time.perf_counter() - start


class LCPipeline:
    def __init__(self, llm, judge_llm, index_dir="./faiss_index"):
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.generator = Generator(llm)
        self.evaluator = Evaluator(judge_llm)
        self.judge = Judge(judge_llm)
        self.consistency = Consistency(llm, self.generator.prompt)
        self.retriever = None
        # accumulate docs so multiple ingest() calls index every file, not just the last
        self.docs = None

    def ingest(self, file_path):
        docs = self.indexer.load_and_split(file_path)
        if self.docs is None:
            self.docs = docs
        else:
            self.docs.extend(docs)
        db = self.datastore.create(self.docs)
        self.retriever = Retriever(db)

    def load_existing(self):
        db = self.datastore.load()
        self.retriever = Retriever(db)

    def query(self, question):
        docs = self.retriever.retrieve(question)
        draft = answer_text(self.generator.query(question, docs))
        self.last_report = self.verify(question, draft, [doc.page_content for doc in docs])
        self.print_report(self.last_report)
        return self.last_report["answer"], docs

    # sees only the question, the draft and the retrieved context, never the reference
    def verify(self, question, draft, chunks):
        spans = split_claims(draft)
        claims = [draft[start:end] for start, end in spans]
        timings = {}
        report = {"draft": draft, "claims": claims, "answer": draft, "timings": timings, "llm_calls": 0}
        # an abstaining draft has nothing to verify
        if not claims or draft.strip() == NOT_FOUND:
            return report

        with timed(timings, "nli"):
            matrix = support_matrix(question, claims, chunks)
        nli_support = [float(score) for score in matrix.max(axis=1)]
        with timed(timings, "judge"):
            judge_support = self.judge.judge(question, claims, chunks)
        if judge_support is None:
            support = nli_support
        else:
            support = [(nli + judged) / 2 for nli, judged in zip(nli_support, judge_support)]
        keep = [score >= CLAIM_SUPPORT_THRESHOLD for score in support]
        aggregate_support = sum(support) / len(support)

        # sampled once and shared by the consistency score and the semantic entropy
        with timed(timings, "sample"):
            answers = [draft] + self.consistency.sample(question, chunks)
        with timed(timings, "entropy"):
            entropy, clusters = semantic_entropy(question, answers)
        with timed(timings, "consistency"):
            consistency = consistency_score(answers)

        reasons = []
        if aggregate_support < SUPPORT_THRESHOLD:
            reasons.append("support")
        if entropy > ENTROPY_THRESHOLD:
            reasons.append("entropy")
        if not any(keep):
            reasons.append("all claims failed")

        if reasons:
            report["answer"], records, coverage = NOT_FOUND, [], None
        else:
            report["answer"] = revise(draft, spans, keep)
            records, coverage = provenance(claims, matrix, keep)

        report.update(
            nli_support=nli_support, judge_support=judge_support, support=support,
            aggregate_support=aggregate_support, kept=keep, dropped=keep.count(False),
            samples=answers[1:], consistency=consistency, entropy=entropy,
            clusters=clusters, abstained=bool(reasons), abstain_reasons=reasons,
            provenance=records, coverage=coverage, llm_calls=1 + SAMPLE_COUNT,
        )
        return report

    def print_report(self, report):
        added = sum(report["timings"].values())
        if "aggregate_support" not in report:
            print(f"Post-processing: draft already abstains, checks skipped llm_calls=+0 added={added:.2f}s")
            return
        abstained = f"yes ({', '.join(report['abstain_reasons'])})" if report["abstained"] else "no"
        coverage = "-" if report["coverage"] is None else f"{report['coverage']:.0f}%"
        judge = "ok" if report["judge_support"] is not None else "unparsed, NLI only"
        breakdown = " ".join(f"{stage} {seconds:.2f}" for stage, seconds in report["timings"].items())
        print(f"Post-processing: claims={len(report['claims'])} dropped={report['dropped']} "
              f"support={report['aggregate_support']:.2f} consistency={report['consistency']:.2f} "
              f"entropy={report['entropy']:.2f} abstained={abstained} coverage={coverage} judge={judge} "
              f"llm_calls=+{report['llm_calls']} (judge 1, samples {SAMPLE_COUNT}) "
              f"added={added:.2f}s [{breakdown}]")

    def evaluate(self, question, reference):
        answer, _ = self.query(question)
        score, claim_scores = self.evaluator.evaluate(question, answer, reference)
        return {
            "answer": answer, "score": score, "claim_scores": claim_scores,
            "verification": self.last_report,
        }
