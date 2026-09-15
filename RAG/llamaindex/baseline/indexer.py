from llama_index.core.node_parser import SentenceSplitter
from .._loaders import load_documents

class Indexer:
  def __init__(self, chunk_size=1000, chunk_overlap=200):
    # sizes are counted in characters (tokenizer=list) like LangChain's splitter;
    # 1000/200 is what the LangChain indexer's load_and_split() actually uses
    self.splitter = SentenceSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        paragraph_separator = "\n\n",
        tokenizer = list
    )

  def load_and_split(self, file_path):
    documents = load_documents(file_path)

    # keep metadata out of the embedded and prompt text, as LangChain does
    for document in documents:
      document.excluded_embed_metadata_keys = list(document.metadata)
      document.excluded_llm_metadata_keys = list(document.metadata)

    nodes = self.splitter.get_nodes_from_documents(documents)

    # LangChain emits no chunk for an empty page; SentenceSplitter emits ""
    return [node for node in nodes if node.get_content().strip()]
