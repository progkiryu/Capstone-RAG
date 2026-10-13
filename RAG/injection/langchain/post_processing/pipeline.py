import time

from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .checks import most_similar_chunk, new_canary
from .detectors import classifier_score, heuristic_flag
from .policy import ALLOWED_DOMAINS, CONDITIONS, DEFAULT_CONDITION, postprocess

# same threshold as the pre_processing classifier (P3/P5)
DETECTOR_THRESHOLD = 0.9

class LCPipeline:
    def __init__(self, llm, judge_llm=None, index_dir="./faiss_index", condition=DEFAULT_CONDITION):
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.canary = new_canary()
        self.generator = Generator(llm, self.canary)
        self.evaluator = Evaluator()
        self.judge_llm = judge_llm
        self.retriever = None
        self.condition = condition
        self.config = CONDITIONS[condition]
        self.last_log = {}
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
        chunks = [doc.page_content for doc in docs]
        response = self.generator.query(question, chunks)

        start = time.perf_counter()
        grounding = self.grounding_chunks(chunks)
        final, log = postprocess(
            question, response, grounding, self.canary, ALLOWED_DOMAINS, self.config,
            call_judge=self.call_judge if self.judge_llm is not None else None,
            regen=lambda flagged: self.regenerate(question, chunks, flagged),
        )
        log["grounding_chunks"] = len(grounding)
        self.last_log = log
        self.report(log, len(chunks), time.perf_counter() - start)
        return final, docs

    def grounding_chunks(self, chunks):
        if not (self.config.grounded and self.config.grounded_chunks == "unflagged"):
            return chunks
        # detectors run after generation, only to choose which chunks may ground the answer
        return [
            chunk for chunk in chunks
            if not heuristic_flag(chunk) and classifier_score(chunk) < DETECTOR_THRESHOLD
        ]

    def call_judge(self, prompt):
        return self.judge_llm.invoke(prompt).text

    def regenerate(self, question, chunks, flagged):
        drop = most_similar_chunk(flagged, chunks)
        return self.generator.query(question, chunks[:drop] + chunks[drop + 1:])

    def report(self, log, chunk_count, seconds):
        calls = int("judge" in log) + int("regenerate" in log["actions"])
        judge = log.get("judge")
        verdict = f"{judge.get('verdict')} ({judge.get('reason')})" if judge else "-"
        print(f"Post-processing [{self.condition}]: actions={log['actions'] or ['accept']} judge={verdict} "
              f"grounding_chunks={log['grounding_chunks']}/{chunk_count} llm_calls=+{calls} "
              f"added={seconds:.2f}s")

    def evaluate(self, prediction, reference):
        return self.evaluator.evaluate(prediction, reference)
