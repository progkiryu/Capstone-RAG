import logging
from llama_index.core.node_parser import SentenceSplitter

# importing llama_index.readers.file runs logging.basicConfig(level=INFO), which prints every
# HTTP request; restore the root logger afterwards so output matches the LangChain pipelines
_root_handlers, _root_level = logging.root.handlers[:], logging.root.level
from llama_index.readers.file import PDFReader
logging.root.handlers = _root_handlers
logging.root.setLevel(_root_level)

class Indexer:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        # sizes are counted in characters (tokenizer=list) like LangChain's splitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # split on paragraphs, then sentences, then words (pre-processing)
            paragraph_separator="\n\n",
            tokenizer=list
        )

    def load_and_split(self, file_path):
        # add metadata (pre-processing)
        documents = PDFReader().load_data(file_path, extra_info={"source": file_path})

        # keep metadata out of the embedded, BM25 and prompt text, as LangChain does
        for document in documents:
            document.excluded_embed_metadata_keys = list(document.metadata)
            document.excluded_llm_metadata_keys = list(document.metadata)

        nodes = self.splitter.get_nodes_from_documents(documents)

        # LangChain emits no chunk for an empty page; SentenceSplitter emits ""
        return [node for node in nodes if node.get_content().strip()]
