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
from .defence import CONDITIONS as SCREEN_CONDITIONS, preprocess
from .faithfulness_judge import Judge
from .injection_judge import judge as injection_judge
from .consistency import SAMPLE_COUNT, Consistency, consistency_score, semantic_entropy
from .checks import canary_leaked, exfil_findings, new_canary, off_task_claims, strip_findings
from .verifier import (
    CLAIM_SUPPORT_THRESHOLD, answer_text, provenance, revise, split_claims, support_matrix
)

# hallucination gate, as in hallucination combined: fixed from the observed support/entropy
# distributions (no reference answers consulted)
SUPPORT_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 1.10
# injection screening uses the pre_processing full-stack (P5) detector settings
SCREEN = SCREEN_CONDITIONS["P5"]
ALIGN_THRESHOLD = 0.2
MIN_CHARS = 20
ALLOWED_DOMAINS = set()


@contextmanager
def timed(timings, stage):
    start = time.perf_counter()
    yield
    timings[stage] = timings.get(stage, 0.0) + time.perf_counter() - start


def chars(nodes):
    return sum(len(node.get_content()) for node in nodes)


class LIPipeline:
    def __init__(self, llm, judge_llm, index_dir="./faiss_index"):
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.canary = new_canary()
        self.generator = Generator(llm, self.canary)
        self.evaluator = Evaluator(judge_llm)
        self.rewriter = Rewriter(llm)
        self.grader = Grader(llm)
        self.nli_filter = NLIFilter()
        self.compressor = Compressor()
        self.judge = Judge(judge_llm)
        self.judge_llm = judge_llm
        self.consistency = Consistency(llm, self.generator)
        self.retriever = None
        # accumulate nodes so multiple ingest() calls index every file; BM25 also needs them
        self.nodes = None

    def ingest(self, file_path):
        nodes = self.indexer.load_and_split(file_path)
        if self.nodes is None:
            self.nodes = nodes
        else:
            self.nodes.extend(nodes)
        index = self.datastore.create(self.nodes)
        self.retriever = Retriever(index, self.nodes, self.datastore.embedding)

    def load_existing(self):
        if self.nodes is None:
            raise ValueError("BM25 needs the chunks: call ingest() before load_existing()")
        index = self.datastore.load()
        self.retriever = Retriever(index, self.nodes, self.datastore.embedding)

    def screen(self, nodes):
        result = preprocess([node.get_content() for node in nodes], SCREEN)
        kept = [entry["chunk"] for entry in result.log if not entry["flagged"]]
        # copies, so the docstore's nodes keep their original text
        screened = [
            nodes[i].model_copy(update={"text": text})
            for i, text in zip(kept, result.chunks)
        ]
        return screened, result.log

    def query(self, question):
        timings = {}
        with timed(timings, "rewrite"):
            search_query, rewritten = self.rewriter.rewrite(question)
        with timed(timings, "retrieve"):
            nodes = self.retriever.retrieve(search_query)
        # screened before the grader, which is an LLM reading the chunks
        with timed(timings, "screen"):
            nodes, screen_log = self.screen(nodes)
        llm_calls, grade, parsed = 1, None, True

        if nodes:
            with timed(timings, "grade"):
                grade, parsed = self.grader.grade(question, nodes)
            llm_calls += 1
            if grade == AMBIGUOUS:
                # one retry with a different rewrite, never re-graded, so this cannot loop
                with timed(timings, "retry"):
                    retry_query, _ = self.rewriter.rewrite(question, previous=search_query)
                    nodes, screen_log = self.screen(self.retriever.retrieve(retry_query))
                llm_calls += 1
        retrieved = nodes

        if not retrieved:
            pre_reason = "all chunks flagged"
        elif grade == INCORRECT:
            pre_reason = "grade INCORRECT"
        else:
            pre_reason = None

        pre = {
            "search_query": search_query, "rewrite_used": rewritten, "grade": grade,
            "grade_parsed": parsed, "retrieved": len(retrieved), "after_nli": None,
            "compressed": None, "chars_retrieved": chars(retrieved), "chars_compressed": None,
            "llm_calls": llm_calls, "timings": timings, "abstain_reason": pre_reason,
        }
        if pre_reason:
            context, draft = retrieved, NOT_FOUND
        else:
            with timed(timings, "nli"):
                filtered = self.nli_filter.filter(question, retrieved)
            with timed(timings, "compress"):
                context = self.compressor.compress(question, filtered)
            pre.update(after_nli=len(filtered), compressed=len(context), chars_compressed=chars(context))
            draft = answer_text(self.generator.query(question, [node.get_content() for node in context]))
        self.print_pre_report(pre)
        self.print_screen_report(screen_log)

        # the post-processing checks verify against the compressed context the generator saw
        self.last_report = self.verify(question, draft, [node.get_content() for node in context])
        self.last_report.update(pre_processing=pre, injection_screen=screen_log)
        self.print_post_report(self.last_report)
        return self.last_report["answer"], context

    # sees only the question, the draft and the retrieved context, never the reference
    def verify(self, question, draft, chunks):
        timings = {}
        report = {"draft": draft, "claims": [], "answer": draft, "timings": timings, "llm_calls": 0}
        # an abstaining draft has nothing to verify
        if draft.strip() == NOT_FOUND:
            return report

        if canary_leaked(draft, self.canary):
            report.update(answer=NOT_FOUND, abstained=True, abstain_reasons=["canary leaked"])
            return report

        with timed(timings, "exfil"):
            findings = exfil_findings(draft, question, ALLOWED_DOMAINS)
        text = strip_findings(draft, findings) if findings else draft
        spans = split_claims(text)
        claims = [text[start:end] for start, end in spans]
        report.update(claims=claims, stripped=findings)
        if not claims:
            report.update(answer=NOT_FOUND, abstained=True, abstain_reasons=["nothing left after stripping"])
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
        with timed(timings, "alignment"):
            off_task = off_task_claims(question, claims, ALIGN_THRESHOLD)
        keep = [score >= CLAIM_SUPPORT_THRESHOLD and not off for score, off in zip(support, off_task)]
        aggregate_support = sum(support) / len(support)

        # sampled once and shared by the consistency score and the semantic entropy
        with timed(timings, "sample"):
            answers = [draft] + self.consistency.sample(question, chunks)
        with timed(timings, "entropy"):
            entropy, clusters = semantic_entropy(question, answers)
        with timed(timings, "consistency"):
            consistency = consistency_score(answers)
        llm_calls = 1 + SAMPLE_COUNT

        reasons = []
        if aggregate_support < SUPPORT_THRESHOLD:
            reasons.append("support")
        if entropy > ENTROPY_THRESHOLD:
            reasons.append("entropy")
        if not any(keep):
            reasons.append("all claims failed")

        verdict = None
        if not reasons:
            answer = revise(text, spans, keep)
            with timed(timings, "injection_judge"):
                verdict = injection_judge(question, answer, self.call_judge)
            llm_calls += 1
            if verdict.get("verdict") != "clean":
                reasons.append("injection judge")
            # only an answer cut down by an injection check counts as too short
            elif (findings or any(off_task)) and len(answer) < MIN_CHARS:
                reasons.append("too short")

        if reasons:
            report["answer"], records, coverage = NOT_FOUND, [], None
        else:
            report["answer"] = answer
            records, coverage = provenance(claims, matrix, keep)

        report.update(
            nli_support=nli_support, judge_support=judge_support, support=support,
            aggregate_support=aggregate_support, off_task=off_task, kept=keep,
            dropped=keep.count(False), samples=answers[1:], consistency=consistency,
            entropy=entropy, clusters=clusters, injection_verdict=verdict,
            abstained=bool(reasons), abstain_reasons=reasons, provenance=records,
            coverage=coverage, llm_calls=llm_calls,
        )
        return report

    def call_judge(self, prompt):
        return self.judge_llm.complete(prompt).text

    def print_pre_report(self, pre):
        # the first retrieval is what baseline also pays; everything else is added by the measures
        added = sum(seconds for stage, seconds in pre["timings"].items() if stage != "retrieve")
        breakdown = " ".join(f"{stage} {seconds:.2f}" for stage, seconds in pre["timings"].items())
        after_nli = pre["after_nli"] if pre["after_nli"] is not None else "-"
        after_compression = (
            f"{pre['compressed']} ({pre['chars_retrieved']}->{pre['chars_compressed']} chars)"
            if pre["compressed"] is not None else "-"
        )
        grade = pre["grade"] or "-"
        abstained = f" abstained=yes ({pre['abstain_reason']})" if pre["abstain_reason"] else ""
        print(f"Pre-processing: retrieved={pre['retrieved']} nli={after_nli} "
              f"compressed={after_compression} grade={grade}"
              f"{'' if pre['grade_parsed'] else ' (unparsed, failed open)'} "
              f"rewrite={'used' if pre['rewrite_used'] else 'rejected'}{abstained} "
              f"llm_calls=+{pre['llm_calls']} added={added:.2f}s [{breakdown}]")

    def print_screen_report(self, log):
        flagged = [entry["chunk"] for entry in log if entry["flagged"]]
        heuristic = sum(bool(entry["heuristic"]) for entry in log)
        classifier = sum(entry["clf_score"] is not None and entry["clf_score"] >= SCREEN.threshold for entry in log)
        print(f"Injection screen: chunks={len(log)} flagged={flagged} "
              f"(heuristic {heuristic}, classifier {classifier}) dropped={len(flagged)}")

    def print_post_report(self, report):
        added = sum(report["timings"].values())
        if "abstain_reasons" not in report:
            print(f"Post-processing: draft already abstains, checks skipped llm_calls=+0 added={added:.2f}s")
            return
        abstained = f"yes ({', '.join(report['abstain_reasons'])})" if report["abstained"] else "no"
        if "aggregate_support" not in report:
            print(f"Post-processing: abstained={abstained} llm_calls=+0 added={added:.2f}s")
            return
        coverage = "-" if report["coverage"] is None else f"{report['coverage']:.0f}%"
        judge = "ok" if report["judge_support"] is not None else "unparsed, NLI only"
        verdict = report["injection_verdict"]
        injection = f"{verdict.get('verdict')} ({verdict.get('reason')})" if verdict else "-"
        breakdown = " ".join(f"{stage} {seconds:.2f}" for stage, seconds in report["timings"].items())
        print(f"Post-processing: claims={len(report['claims'])} dropped={report['dropped']} "
              f"off_task={sum(report['off_task'])} links_stripped={len(report['stripped'])} "
              f"support={report['aggregate_support']:.2f} consistency={report['consistency']:.2f} "
              f"entropy={report['entropy']:.2f} injection_judge={injection} abstained={abstained} "
              f"coverage={coverage} faithfulness_judge={judge} llm_calls=+{report['llm_calls']} "
              f"added={added:.2f}s [{breakdown}]")

    def evaluate(self, question, reference):
        answer, _ = self.query(question)
        score, claim_scores = self.evaluator.evaluate(question, answer, reference)
        return {
            "answer": answer, "score": score, "claim_scores": claim_scores,
            "verification": self.last_report,
        }
