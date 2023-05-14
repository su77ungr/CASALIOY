"""ingest documents into vector database using embedding"""
import os
import shutil
import sys
from hashlib import md5
from pathlib import Path

from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredEPubLoader, \
    UnstructuredHTMLLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from load_env import persist_directory, chunk_size, chunk_overlap, llama_embeddings_model, model_n_ctx, \
    documents_directory, use_mlock

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


def embed_documents_with_progress(embedding_model: LlamaCppEmbeddings, texts: list[str]) -> list[list[float]]:
    """wrapper around embed_documents that prints progress"""
    embeddings = []
    N_chunks = len(texts)
    for i, text in enumerate(texts):
        print(f"embedding chunk {i+1}/{N_chunks}")
        embeddings.append(embedding_model.client.embed(text))

    return [list(map(float, e)) for e in embeddings]


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

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(documents)
    texts = [d.page_content for d in split_documents]
    metadatas = [d.metadata for d in split_documents]
    print(f"Found {len(split_documents)} chunks from {len(documents)} documents to index")

    # Generate embeddings
    print("Generating embeddings...")
    embedding_model = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx, use_mlock=use_mlock)
    embeddings = embed_documents_with_progress(embedding_model, texts)

    # Store embeddings
    print("Storing embeddings...")
    client = QdrantClient(path=db_dir)  # using Qdrant.from_documents recreates the db each time
    try:
        collection = client.get_collection("test")
    except ValueError:  # doesn't exist
        print("Creating a new store")
        # Just do a single quick embedding to get vector size
        vector_size = max(len(e) for e in embeddings)
        client.recreate_collection(
            collection_name="test",
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance["COSINE"],
            ),
        )
        collection = client.get_collection("test")
    print(f"Loaded collection has {collection.points_count} data points")
    client.upsert(
        collection_name="test",
        points=models.Batch.construct(
            ids=[md5(text.encode("utf-8")).hexdigest() for text in texts],
            vectors=embeddings,
            payloads=[{"page_content": text, "metadata": metadatas[i]} for i, text in enumerate(texts)]
        ),
    )
    collection = client.get_collection("test")
    print(f"Indexed {len(split_documents)} chunks from {len(documents)} documents in Qdrant. Total points: {collection.points_count}")


if __name__ == "__main__":
    sources_directory = sys.argv[1] if len(sys.argv) > 1 else documents_directory
    cleandb = sys.argv[2] if len(sys.argv) > 2 else "n"
    main(sources_directory, cleandb)
