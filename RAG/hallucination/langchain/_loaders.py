import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(file_path):
    """Load a file into LangChain Documents, dispatching on extension.

    PDFs go through PyPDFLoader (one Document per page). Everything else
    (.txt, .html, .md, ...) is read as RAW text via TextLoader -- HTML is NOT
    stripped, so hidden payloads (comments, display:none spans) survive for the
    injection tests.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()

    return TextLoader(file_path, encoding="utf-8").load()
