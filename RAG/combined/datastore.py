# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import OPENAI_API_KEY

class Datastore:
    def __init__(self, persist_dir="./faiss_index"):
        self.embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        self.persist_dir = persist_dir
        self.score_threshold = 0.75
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

    def search(self, query, k=5):
        if self.db is None:
            raise ValueError("Database not loaded")

        # FAISS similarity search returns (pre-processing)
        results = self.db.similarity_search_with_score(query, k=k)

        filtered = []
        for doc, score in results:
            similarity = 1 - score if score <= 1 else 1 / (1 + score)

            if similarity >= self.score_threshold:
                filtered.append({
                    "document": doc,
                    "score": similarity
                })
        return filtered