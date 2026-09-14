from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import GEMINI_API_KEY, OPENAI_API_KEY

class Datastore:
    def __init__(self, persist_dir="./faiss_index"):
        self.embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        self.persist_dir = persist_dir
        self.db = None

    def create(self, documents):
        self.db = FAISS.from_documents(
            documents,
            embedding=self.embedding
        )
        self.db.save_local(self.persist_dir)
        return self.db

    def load(self):
        self.db = FAISS.load_local(
            self.persist_dir,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True
        )
        return self.db