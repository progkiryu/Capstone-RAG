import os
import logging

from llama_index.core import Document

# importing llama_index.readers.file runs logging.basicConfig(level=INFO), which prints every
# HTTP request; restore the root logger afterwards so output matches the other pipelines
_root_handlers, _root_level = logging.root.handlers[:], logging.root.level
from llama_index.readers.file import PDFReader
logging.root.handlers = _root_handlers
logging.root.setLevel(_root_level)


def load_documents(file_path, extra_info=None):
    """Load a file into LlamaIndex Documents, dispatching on extension.

    PDFs go through PDFReader (one Document per page). Everything else
    (.txt, .html, .md, ...) is read as RAW text -- HTML is NOT stripped, so
    hidden payloads (comments, display:none spans) survive for the injection
    tests.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PDFReader().load_data(file_path, extra_info=extra_info)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [Document(text=text, metadata=extra_info or {})]
