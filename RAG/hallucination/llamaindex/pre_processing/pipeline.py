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


def chars(nodes):
    return sum(len(node.get_content()) for node in nodes)


class LIPipeline:
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

    def query(self, question):
        timings = {}
        with timed(timings, "rewrite"):
            search_query, rewritten = self.rewriter.rewrite(question)
        with timed(timings, "retrieve"):
            nodes = self.retriever.retrieve(search_query)
        with timed(timings, "grade"):
            grade, parsed = self.grader.grade(question, nodes)
        llm_calls = 2

        if grade == AMBIGUOUS:
            # one retry with a different rewrite, never re-graded, so this cannot loop
            with timed(timings, "retry"):
                retry_query, _ = self.rewriter.rewrite(question, previous=search_query)
                nodes = self.retriever.retrieve(retry_query)
            llm_calls += 1
        retrieved = nodes

        if grade == INCORRECT:
            self.report(retrieved, None, None, grade, parsed, rewritten, llm_calls, timings)
            return NOT_FOUND, nodes

        with timed(timings, "nli"):
            filtered = self.nli_filter.filter(question, retrieved)
        with timed(timings, "compress"):
            nodes = self.compressor.compress(question, filtered)
        self.report(retrieved, filtered, nodes, grade, parsed, rewritten, llm_calls, timings)

        response = self.generator.query(question, nodes)
        return response, nodes

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
