from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re
import os

class Indexer:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # separators to try when splitting documents (pre-processing)
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def load_and_split(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        docs = self.splitter.split_documents(documents)

        # add metadata (pre-processing)
        for doc in docs:
            doc.metadata["source"] = file_path

        return docs