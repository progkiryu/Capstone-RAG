import time
from contextlib import contextmanager

from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .rewriter import Rewriter
from .grader import Grader, AMBIGUOUS, INCORRECT, NOT_FOUND
from .filter import NLIFilter
from .compressor import Compressor
from .judge import Judge
from .consistency import SAMPLE_COUNT, Consistency, consistency_score, semantic_entropy
from .verifier import (
    CLAIM_SUPPORT_THRESHOLD, answer_text, provenance, revise, split_claims, support_matrix
)

# same gate as post_processing, fixed from the observed support/entropy distributions
# (no reference answers consulted): abstain when claims are on average unsupported, or when
# the sampled answers are spread over more meanings than an even three-way split (ln 3)
SUPPORT_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 1.10


@contextmanager
def timed(timings, stage):
    start = time.perf_counter()
    yield
    timings[stage] = timings.get(stage, 0.0) + time.perf_counter() - start


def chars(docs):
    return sum(len(doc.page_content) for doc in docs)


class LCPipeline:
    def __init__(self, llm, judge_llm, index_dir="./faiss_index"):
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.generator = Generator(llm)
        self.evaluator = Evaluator(judge_llm)
        self.rewriter = Rewriter(llm)
        self.grader = Grader(llm)
        self.nli_filter = NLIFilter()
        self.compressor = Compressor()
        self.judge = Judge(judge_llm)
        self.consistency = Consistency(llm, self.generator.prompt)
        self.retriever = None
        # accumulate docs so multiple ingest() calls index every file; BM25 also needs them
        self.docs = None

    def ingest(self, file_path):
        docs = self.indexer.load_and_split(file_path)
        if self.docs is None:
            self.docs = docs
        else:
            self.docs.extend(docs)
        db = self.datastore.create(self.docs)
        self.retriever = Retriever(db, self.docs)

    def load_existing(self):
        if self.docs is None:
            raise ValueError("BM25 needs the chunks: call ingest() before load_existing()")
        db = self.datastore.load()
        self.retriever = Retriever(db, self.docs)

    def query(self, question):
        timings = {}
        with timed(timings, "rewrite"):
            search_query, rewritten = self.rewriter.rewrite(question)
        with timed(timings, "retrieve"):
            docs = self.retriever.retrieve(search_query)
        with timed(timings, "grade"):
            grade, parsed = self.grader.grade(question, docs)
        llm_calls = 2

        if grade == AMBIGUOUS:
            # one retry with a different rewrite, never re-graded, so this cannot loop
            with timed(timings, "retry"):
                retry_query, _ = self.rewriter.rewrite(question, previous=search_query)
                docs = self.retriever.retrieve(retry_query)
            llm_calls += 1
        retrieved = docs

        pre = {
            "search_query": search_query, "rewrite_used": rewritten, "grade": grade,
            "grade_parsed": parsed, "retrieved": len(retrieved), "after_nli": None,
            "compressed": None, "chars_retrieved": chars(retrieved), "chars_compressed": None,
            "llm_calls": llm_calls, "timings": timings,
        }
        if grade == INCORRECT:
            context, draft = retrieved, NOT_FOUND
        else:
            with timed(timings, "nli"):
                filtered = self.nli_filter.filter(question, retrieved)
            with timed(timings, "compress"):
                context = self.compressor.compress(question, filtered)
            pre.update(after_nli=len(filtered), compressed=len(context), chars_compressed=chars(context))
            draft = answer_text(self.generator.query(question, context))
        self.print_pre_report(pre)

        # the post-processing checks verify against the compressed context the generator saw
        self.last_report = self.verify(question, draft, [doc.page_content for doc in context])
        self.last_report["pre_processing"] = pre
        self.print_post_report(self.last_report)
        return self.last_report["answer"], context

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

    def print_pre_report(self, pre):
        # the first retrieval is what baseline also pays; everything else is added by the measures
        added = sum(seconds for stage, seconds in pre["timings"].items() if stage != "retrieve")
        breakdown = " ".join(f"{stage} {seconds:.2f}" for stage, seconds in pre["timings"].items())
        after_nli = pre["after_nli"] if pre["after_nli"] is not None else "-"
        after_compression = (
            f"{pre['compressed']} ({pre['chars_retrieved']}->{pre['chars_compressed']} chars)"
            if pre["compressed"] is not None else "-"
        )
        print(f"Pre-processing: retrieved={pre['retrieved']} nli={after_nli} "
              f"compressed={after_compression} grade={pre['grade']}"
              f"{'' if pre['grade_parsed'] else ' (unparsed, failed open)'} "
              f"rewrite={'used' if pre['rewrite_used'] else 'rejected'} llm_calls=+{pre['llm_calls']} "
              f"added={added:.2f}s [{breakdown}]")

    def print_post_report(self, report):
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
