import time
from contextlib import contextmanager

from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .filter import NLIFilter
from .compressor import Compressor
from .rewriter import Rewriter
from .grader import Grader, AMBIGUOUS, INCORRECT, NOT_FOUND


@contextmanager
def timed(timings, stage):
    start = time.perf_counter()
    yield
    timings[stage] = timings.get(stage, 0.0) + time.perf_counter() - start


def chars(docs):
    return sum(len(doc.page_content) for doc in docs)


class LCPipeline:
    def __init__(self, llm, judge_llm=None, index_dir="./faiss_index"):
        # judge_llm unused: pre_processing uses heuristic scoring
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.generator = Generator(llm)
        self.evaluator = Evaluator()
        self.rewriter = Rewriter(llm)
        self.grader = Grader(llm)
        self.nli_filter = NLIFilter()
        self.compressor = Compressor()
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

        if grade == INCORRECT:
            self.report(retrieved, None, None, grade, parsed, rewritten, llm_calls, timings)
            return NOT_FOUND, docs

        with timed(timings, "nli"):
            filtered = self.nli_filter.filter(question, retrieved)
        with timed(timings, "compress"):
            docs = self.compressor.compress(question, filtered)
        self.report(retrieved, filtered, docs, grade, parsed, rewritten, llm_calls, timings)

        response = self.generator.query(question, docs)
        return response, docs

    def report(self, retrieved, filtered, compressed, grade, parsed, rewritten, llm_calls, timings):
        # the first retrieval is what baseline also pays; everything else is added by the measures
        added = sum(seconds for stage, seconds in timings.items() if stage != "retrieve")
        breakdown = " ".join(f"{stage} {seconds:.2f}" for stage, seconds in timings.items())
        after_nli = len(filtered) if filtered is not None else "-"
        after_compression = (
            f"{len(compressed)} ({chars(retrieved)}->{chars(compressed)} chars)"
            if compressed is not None else "-"
        )
        print(f"Pre-processing: retrieved={len(retrieved)} nli={after_nli} "
              f"compressed={after_compression} grade={grade}{'' if parsed else ' (unparsed, failed open)'} "
              f"rewrite={'used' if rewritten else 'rejected'} llm_calls=+{llm_calls} "
              f"added={added:.2f}s [{breakdown}]")

    def evaluate(self, prediction, reference):
        return self.evaluator.evaluate(prediction, reference)
