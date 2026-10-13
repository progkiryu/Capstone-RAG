import time

from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .defence import CONDITIONS, DEFAULT_CONDITION, FALLBACK, preprocess

class LIPipeline:
    def __init__(self, llm, judge_llm=None, index_dir="./faiss_index", condition=DEFAULT_CONDITION):
        # judge_llm unused: the injection test scores by marker matching
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.generator = Generator(llm)
        self.evaluator = Evaluator()
        self.retriever = None
        self.condition = condition
        self.config = CONDITIONS[condition]
        self.last_log = []
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
        start = time.perf_counter()
        result = preprocess([node.get_content() for node in nodes], self.config)
        self.last_log = result.log
        fallback = all(entry["flagged"] for entry in result.log)
        self.report(result.log, fallback, time.perf_counter() - start)
        if fallback:
            # every chunk was flagged, so skip the generator rather than answer from no context
            return FALLBACK, nodes
        response = self.generator.query(question, result.chunks, self.config.safe_prompt, self.config.spotlight)
        return response, nodes

    def report(self, log, fallback, seconds):
        flagged = [entry["chunk"] for entry in log if entry["flagged"]]
        heuristic = sum(bool(entry["heuristic"]) for entry in log)
        classifier = sum(
            entry["clf_score"] is not None and entry["clf_score"] >= self.config.threshold for entry in log
        )
        action = "replaced" if self.config.replace_instead_of_drop else "dropped"
        print(f"Pre-processing [{self.condition}]: chunks={len(log)} flagged={flagged} "
              f"(heuristic {heuristic}, classifier {classifier}) {action}={len(flagged)} "
              f"fallback={'yes' if fallback else 'no'} added={seconds:.2f}s")

    def evaluate(self, prediction, reference):
        return self.evaluator.evaluate(prediction, reference)
