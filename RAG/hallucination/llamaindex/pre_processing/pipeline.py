from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .filter import NLIFilter
from .compressor import Compressor
from .rewriter import Rewriter
from .grader import Grader, AMBIGUOUS, INCORRECT, NOT_FOUND

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
        search_query, _ = self.rewriter.rewrite(question)
        nodes = self.retriever.retrieve(search_query)

        grade, _ = self.grader.grade(question, nodes)
        if grade == AMBIGUOUS:
            # one retry with a different rewrite, never re-graded, so this cannot loop
            retry_query, _ = self.rewriter.rewrite(question, previous=search_query)
            nodes = self.retriever.retrieve(retry_query)
        if grade == INCORRECT:
            return NOT_FOUND, nodes

        nodes = self.nli_filter.filter(question, nodes)
        nodes = self.compressor.compress(question, nodes)
        response = self.generator.query(question, nodes)
        return response, nodes

    def evaluate(self, prediction, reference):
        return self.evaluator.evaluate(prediction, reference)
