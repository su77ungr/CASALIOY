"""ingest documents into vector database using embedding"""
import os
import shutil
import sys
from hashlib import md5
from pathlib import Path
from typing import Callable

import numpy as np
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_env import chunk_overlap, chunk_size, documents_directory, get_embedding_model, persist_directory
from qdrant_client import QdrantClient, models


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
    }

    def __init__(self, db_dir: str, collection: str = "test"):
        self.db_dir = db_dir
        self.collection = collection
        self.save_N = 1000  # save after creating this many embeddings
        self.awaiting_save: list[tuple[list[np.ndarray], Document]] = []  # [(embedding, document)]

    def load_one_doc(self, filepath: Path) -> list[Document]:
        """load one document"""
        if filepath.suffix[1:] not in self.file_loaders:
            print(f"Unhandled file format: {filepath.name} in {filepath.parent}")
            return []

        return self.file_loaders[filepath.suffix[1:]](str(filepath)).load()

    def embed_documents_with_progress(self, embedding_function: Callable, documents: list[Document]) -> None:
        """wraps around embed_documents and saves"""
        N_chunks = len(documents)
        print(f"Processing {N_chunks} chunks")

        for i in range(0, len(documents), self.save_N):
            documents_sub = documents[i : i + self.save_N]
            embeddings = embedding_function([doc.page_content for doc in documents_sub]).tolist()
            if i > 0:
                print(f"embedding chunk {i + 1}/{N_chunks}")
            self.awaiting_save += list(zip(embeddings, documents_sub))
            if len(self.awaiting_save) >= self.save_N:
                self.store_embeddings()

        self.store_embeddings()

    def store_embeddings(self) -> None:
        """store embeddings in vector store"""
        if len(self.awaiting_save) == 0:
            return

        client = QdrantClient(path=self.db_dir, prefer_grpc=True)
        try:
            client.get_collection(self.collection)
        except ValueError:  # doesn't exist
            # Just do a single quick embedding to get vector size
            vector_size = max(len(e[0]) for e in self.awaiting_save)
            print(f"Creating a new collection, size={vector_size}")
            client.recreate_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance["COSINE"],
                ),
            )

        print(f"Saving {len(self.awaiting_save)} chunks")
        embeddings, texts, metadatas = (
            [e[0] for e in self.awaiting_save],
            [e[1].page_content for e in self.awaiting_save],
            [e[1].metadata for e in self.awaiting_save],
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
        self.awaiting_save = []
        print(f"Saved, the collection now holds {collection.points_count} documents.")

    def ingest_from_directory(self, path: str, chunk_size: int, chunk_overlap: int) -> None:
        """ingest all supported files from the directory"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        encode_fun = get_embedding_model()[1]

        # get all documents
        print("Scanning files")
        for root, dirs, files in os.walk(path):
            for file in files:
                print(f"Processing {file}")
                document = self.load_one_doc(Path(root) / file)
                split_document = text_splitter.split_documents(document)
                self.embed_documents_with_progress(encode_fun, split_document)
                print(f"Processed {file}")
        print("Done")


def main(sources_directory: str, cleandb: str) -> None:
    """main function"""
    ingester = Ingester(persist_directory)

    if os.path.exists(ingester.db_dir):
        if cleandb.lower() == "y" or (cleandb == "n" and input("\nDelete current database?(Y/N): ").lower() == "y"):
            print("Deleting db...")
            shutil.rmtree(ingester.db_dir)
        elif cleandb.lower() == "n":
            print("Adding to db...")

    ingester.ingest_from_directory(sources_directory, chunk_size, chunk_overlap)


if __name__ == "__main__":
    sources_directory = sys.argv[1] if len(sys.argv) > 1 else documents_directory
    cleandb = sys.argv[2] if len(sys.argv) > 2 else "n"
    main(sources_directory, cleandb)
