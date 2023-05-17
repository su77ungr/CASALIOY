"""ingest documents into vector database using embedding"""
import multiprocessing
import os
import shutil
import sys
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import Callable

from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_env import chunk_overlap, chunk_size, documents_directory, get_embedding_model, ingest_n_threads, persist_directory
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import ProgressBar
from qdrant_client import QdrantClient, models

from casalioy.utils import print_HTML, prompt_HTML


class Ingester:
    """ingest documents"""

    file_loaders = {  # extension -> loader
        "txt": lambda path: TextLoader(path, encoding="utf8"),
        "pdf": PDFMinerLoader,
        "csv": CSVLoader,
        "epub": UnstructuredEPubLoader,
        "html": UnstructuredHTMLLoader,
        "docx": Docx2txtLoader,
        "pptx": UnstructuredPowerPointLoader,
        "eml": UnstructuredEmailLoader,
        "msg": OutlookMessageLoader,
    }

    def __init__(self, db_dir: str, collection: str = "test", verbose=False):
        self.lock: multiprocessing.Lock = None
        self.n_threads = ingest_n_threads
        self.encode_fun = None
        self.text_splitter = None
        self.db_dir = db_dir
        self.collection = collection
        self.verbose = verbose
        self.storing = False

    def load_one_doc(self, filepath: Path) -> list[Document]:
        """load one document"""
        if self.verbose:
            print_HTML("<r>Processing {fname}</r>", fname=filepath.name)
        if filepath.suffix[1:] not in self.file_loaders:
            if self.verbose:
                print_HTML("<w>Unhandled file format: {fname} in {fparent}</w>", fname=filepath.name, fparent=filepath.parent)
            return []

        return self.file_loaders[filepath.suffix[1:]](str(filepath)).load()

    def embed_documents_with_progress(self, embedding_function: Callable, documents: list[Document]) -> None:
        """wraps around embed_documents and saves"""
        if self.verbose:
            print_HTML(f"<r>Processing {len(documents)} chunks</r>")

        embeddings = embedding_function([doc.page_content for doc in documents]).tolist()
        self.store_embeddings(list(zip(embeddings, documents)))

    def store_embeddings(self, embeddings: list) -> None:
        """store embeddings in vector store"""
        with self.lock or multiprocessing.Lock():
            client = QdrantClient(path=self.db_dir, prefer_grpc=True)
            try:
                client.get_collection(self.collection)
            except ValueError:  # doesn't exist
                # Just do a single quick embedding to get vector size
                vector_size = max(len(e[0]) for e in embeddings)
                print_HTML(f"<r>Creating a new collection, size={vector_size}</r>")
                client.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance["COSINE"],
                    ),
                )

            print_HTML(f"<r>Saving {len(embeddings)} chunks</r>")
            embeddings, texts, metadatas = (
                [e[0] for e in embeddings],
                [e[1].page_content for e in embeddings],
                [e[1].metadata for e in embeddings],
            )
            client.upsert(
                collection_name=self.collection,
                points=models.Batch.construct(
                    ids=[md5(text.encode("utf-8")).hexdigest() for text in texts],
                    vectors=embeddings,
                    payloads=[{"page_content": text, "metadata": metadatas[i]} for i, text in enumerate(texts)],
                ),
            )
            collection = client.get_collection(self.collection)
            if self.verbose:
                print_HTML(f"<r>Saved, the collection now holds {collection.points_count} documents.</r>")

    def process_one_doc(self, lock: multiprocessing.Lock, filepath: Path) -> None:
        """process one doc"""
        self.lock = lock
        document = self.load_one_doc(filepath)
        if not document:
            return
        split_document = self.text_splitter.split_documents(document)
        self.embed_documents_with_progress(self.encode_fun, split_document)
        if self.verbose:
            print_HTML("<r>Processed {fname}</r>", fname=filepath.name)

    def ingest_from_directory(self, path: str, chunk_size: int, chunk_overlap: int) -> None:
        """ingest all supported files from the directory"""
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.encode_fun = get_embedding_model()[1]

        # get all documents
        print_HTML("<r>Scanning files</r>")
        all_items = [Path(root) / file for root, dirs, files in os.walk(path) for file in files]
        with ProgressBar() as pb:
            multiprocessing.set_start_method("spawn")
            with multiprocessing.Pool(self.n_threads) as pool:
                lock = multiprocessing.Manager().Lock()
                for _ in pb(pool.imap_unordered(partial(self.process_one_doc, lock), all_items), total=len(all_items)):
                    pass
        print_HTML("<r>Done</r>")


def main(sources_directory: str, cleandb: str) -> None:
    """main function"""
    ingester = Ingester(persist_directory)
    session = PromptSession()

    if os.path.exists(ingester.db_dir):
        if cleandb.lower() == "y" or (cleandb == "n" and prompt_HTML(session, "\n<b><w>Delete current database?(Y/N)</w></b>: ").lower() == "y"):
            print_HTML("<r>Deleting db...</r>")
            shutil.rmtree(ingester.db_dir)
        elif cleandb.lower() == "n":
            print_HTML("<r>Adding to db...</r>")

    ingester.ingest_from_directory(sources_directory, chunk_size, chunk_overlap)


if __name__ == "__main__":
    sources_directory = sys.argv[1] if len(sys.argv) > 1 else documents_directory
    cleandb = sys.argv[2] if len(sys.argv) > 2 else "n"
    main(sources_directory, cleandb)
