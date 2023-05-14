"""ingest documents into vector database using embedding"""
import os
import shutil
import sys

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredEPubLoader, UnstructuredHTMLLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant

load_dotenv()
llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
documents_directory = os.environ.get("DOCUMENTS_DIRECTORY")
model_n_ctx = os.environ.get("MODEL_N_CTX")


def load_one_doc(filepath: str) -> list[Document]:
    """load one document"""
    if filepath.endswith(".txt"):
        loader = TextLoader(filepath, encoding="utf8")
    elif filepath.endswith(".pdf"):
        loader = PDFMinerLoader(filepath)
    elif filepath.endswith(".csv"):
        loader = CSVLoader(filepath)
    elif filepath.endswith(".epub"):
        loader = UnstructuredEPubLoader(filepath)
    elif filepath.endswith(".html"):
        loader = UnstructuredHTMLLoader(filepath)
    else:
        raise ValueError(f"Unhandled file format: .{filepath.split('.')[-1]} in {filepath}")

    return loader.load()


def main(sources_directory: str, cleandb: str) -> None:
    """enables to run python random_path/ to ingest // or 'python random_path/ y' to purge existing db"""
    db_dir = persist_directory  # can be changed to ":memory:" but is not persistant
    if os.path.exists(db_dir):
        if cleandb.lower() == "y" or (cleandb == "n" and input("\nDelete current database?(Y/N): ").lower() == "y"):
            print("Deleting db...")
            shutil.rmtree(db_dir)
        elif cleandb.lower() == "n":
            print("Adding to db...")

    documents = []
    for root, dirs, files in os.walk(sources_directory):
        for file in files:
            documents += load_one_doc(os.path.join(root, file))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    qdrant = Qdrant.from_documents(texts, llama, path=db_dir, collection_name="test")
    qdrant = None
    print(f"Indexed {len(texts)} chunks from {len(documents)} documents in Qdrant")


if __name__ == "__main__":
    sources_directory = sys.argv[1] if len(sys.argv) > 1 else documents_directory
    cleandb = sys.argv[2] if len(sys.argv) > 2 else "n"
    main(sources_directory, cleandb)
