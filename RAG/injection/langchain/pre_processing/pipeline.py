import time

from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .defence import CONDITIONS, DEFAULT_CONDITION, FALLBACK, preprocess

class LCPipeline:
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
        start = time.perf_counter()
        result = preprocess([doc.page_content for doc in docs], self.config)
        self.last_log = result.log
        fallback = all(entry["flagged"] for entry in result.log)
        self.report(result.log, fallback, time.perf_counter() - start)
        if fallback:
            # every chunk was flagged, so skip the generator rather than answer from no context
            return FALLBACK, docs
        response = self.generator.query(question, result.chunks, self.config.safe_prompt, self.config.spotlight)
        return response, docs

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
