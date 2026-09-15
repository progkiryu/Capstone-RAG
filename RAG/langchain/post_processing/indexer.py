from langchain_text_splitters import RecursiveCharacterTextSplitter
from .._loaders import load_documents

class Indexer:
  def __init__(self, chunk_size=500, chunk_overlap=50):
    self.splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

  def load_and_split(self, file_path):
    documents = load_documents(file_path)

    splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200
    )

    docs = splitter.split_documents(documents)
    return docs