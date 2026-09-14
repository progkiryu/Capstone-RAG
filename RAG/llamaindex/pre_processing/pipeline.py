from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever


class LIPipeline:

    def __init__(self, llm, judge_llm=None, index_dir="./faiss_index"):
        # judge_llm unused: pre_processing uses heuristic scoring

        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)

        self.generator = Generator(llm)
        self.evaluator = Evaluator()

        self.retriever = None

        # store nodes for BM25
        self.nodes = None

    def ingest(self, file_path):

        nodes = self.indexer.load_and_split(file_path)

        # store nodes for BM25 retriever
        if self.nodes is None:
            self.nodes = nodes
        else:
            self.nodes.extend(nodes)

        # create / update vector store
        index = self.datastore.create(self.nodes)

        self.retriever = Retriever(index, self.nodes, self.datastore.embedding)
    def load_existing(self):
        index = self.datastore.load()
        if self.nodes is None:
            raise ValueError(
                "BM25 retriever requires nodes. "
                "Load or rebuild nodes before using load_existing()."
            )

        self.retriever = Retriever(index, self.nodes, self.datastore.embedding)
    def query(self, question):

        nodes = self.retriever.retrieve(question)

        response = self.generator.query(question, nodes)

        return response, nodes

    def evaluate(self, prediction, reference):

        return self.evaluator.evaluate(
            prediction,
            reference,
        )
