from pathlib import Path
from typing import Any, List, Optional

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDINGS_MODEL = "text-embedding-3-small"
DOC_PATH = Path(__file__).parent / "intel_documents"


# rag doc
def load_split_rag_docs_from_root(
    root_path: Path,
    encoding: Optional[str] = None,
    chunk_size: int = 500,
    file_extension: str = ".txt",
) -> List[Any]:
    """
    Recursively load and split all text documents from the given root path.

    Args:
        root_path (Path): Root directory containing documents.
        encoding (Optional[str]): Encoding to use for file reading.
        chunk_size (int): Number of characters per chunk.
        file_extension (str): Filter by file extension (default ".txt").

    Returns:
        List of split documents.
    """
    splitter = CharacterTextSplitter(chunk_size=chunk_size)
    split_docs = []

    for file_path in root_path.rglob(f"*{file_extension}"):
        try:
            if encoding:
                raw_docs = TextLoader(file_path, encoding=encoding).load()
            else:
                raw_docs = TextLoader(file_path, autodetect_encoding=True).load()
            split_docs.extend(splitter.split_documents(raw_docs))
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    return split_docs


# create embedding_model and store in FAISS


def create_faiss_retriever(docs: Any) -> Any:

    embedding_model = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

    database = FAISS.from_documents(docs, embedding_model)

    return database.as_retriever()


rag_docs = load_split_rag_docs_from_root(DOC_PATH)

faiss_retriever = create_faiss_retriever(rag_docs)
