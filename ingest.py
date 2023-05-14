"""ingest documents into vector database using embedding"""
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredEPubLoader, \
    UnstructuredHTMLLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant

load_dotenv()
llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
documents_directory = os.environ.get("DOCUMENTS_DIRECTORY")
model_n_ctx = os.environ.get("MODEL_N_CTX")

file_loaders = {  # extension -> loader
    "txt": lambda path: TextLoader(path, encoding="utf8"),
    "pdf": PDFMinerLoader,
    "csv": CSVLoader,
    "epub": UnstructuredEPubLoader,
    "html": UnstructuredHTMLLoader,
    "docx": Docx2txtLoader,
    "pptx": UnstructuredPowerPointLoader,
}


def load_one_doc(filepath: Path) -> list[Document]:
    """load one document"""
    if filepath.suffix[1:] not in file_loaders:
        print(f"Unhandled file format: {filepath.name} in {filepath.parent}")
        return []

    return file_loaders[filepath.suffix[1:]](str(filepath)).load()


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
            documents += load_one_doc(Path(root) / file)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Found {len(texts)} chunks from {len(documents)} documents to index")
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    Qdrant.from_documents(texts, llama, path=db_dir, collection_name="test")
    print(f"Indexed {len(texts)} chunks from {len(documents)} documents in Qdrant")


if __name__ == "__main__":
    sources_directory = sys.argv[1] if len(sys.argv) > 1 else documents_directory
    cleandb = sys.argv[2] if len(sys.argv) > 2 else "n"
    main(sources_directory, cleandb)
