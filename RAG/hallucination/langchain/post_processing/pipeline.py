from .indexer import Indexer
from .datastore import Datastore
from .generator import Generator
from .evaluator import Evaluator
from .retriever import Retriever
from .verifier import answer_text, split_claims, support_matrix

class LCPipeline:
    def __init__(self, llm, judge_llm, index_dir="./faiss_index"):
        self.indexer = Indexer()
        self.datastore = Datastore(persist_dir=index_dir)
        self.generator = Generator(llm)
        self.evaluator = Evaluator(judge_llm)
        self.retriever = None
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
        draft = answer_text(self.generator.query(question, docs))
        self.last_report = self.verify(question, draft, [doc.page_content for doc in docs])
        return self.last_report["answer"], docs

    # sees only the question, the draft and the retrieved context, never the reference
    def verify(self, question, draft, chunks):
        claims = [draft[start:end] for start, end in split_claims(draft)]
        nli = support_matrix(question, claims, chunks).max(axis=1) if claims else []
        return {"answer": draft, "draft": draft, "claims": claims, "nli_support": [float(score) for score in nli]}

    def evaluate(self, question, reference):
        answer, _ = self.query(question)
        score, claim_scores = self.evaluator.evaluate(question, answer, reference)
        return {"answer": answer, "score": score, "claim_scores": claim_scores}
