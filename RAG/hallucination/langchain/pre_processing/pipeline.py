from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever

class LCPipeline:
    def __init__(self, llm, judge_llm=None, index_dir="./faiss_index"):
        # judge_llm unused: pre_processing uses heuristic scoring
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.generator = Generator(llm)
        self.evaluator = Evaluator()
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
        docs = self.retriever.retrieve(question)
        response = self.generator.query(question, docs)
        return response, docs

    def evaluate(self, prediction, reference):
        return self.evaluator.evaluate(prediction, reference)
