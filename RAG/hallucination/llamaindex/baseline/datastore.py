import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from config import OPENAI_API_KEY

# output size of text-embedding-3-small
EMBEDDING_DIM = 1536

class Datastore:
    def __init__(self, persist_dir="./faiss_index"):
        self.embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        self.persist_dir = persist_dir
        self.db = None

    def create(self, nodes):
        # IndexFlatL2 is the index LangChain's FAISS.from_documents builds
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(EMBEDDING_DIM))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.db = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embedding
        )
        self.db.storage_context.persist(persist_dir=self.persist_dir)
        return self.db

    def load(self):
        vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=self.persist_dir
        )
        self.db = load_index_from_storage(
            storage_context,
            embed_model=self.embedding
        )
        return self.db
