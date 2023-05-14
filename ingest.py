import os
import shutil
import sys

from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredEPubLoader, UnstructuredHTMLLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant

load_dotenv()
llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
documents_directory = os.environ.get("DOCUMENTS_DIRECTORY")
model_n_ctx = os.environ.get("MODEL_N_CTX")


## enables to run python random_path/ to ingest // or 'python random_path/ y' to purge existing db
def main(sources_directory, cleandb):
    db_dir = persist_directory  # can be changed to ":memory:" but is not persistant
    if os.path.exists(db_dir):
        if cleandb.lower() == "y" or (cleandb == "n" and input("\nDelete current database?(Y/N): ").lower() == "y"):
            print("Deleting db...")
            shutil.rmtree(db_dir)
        elif cleandb.lower() == "n":
            print("Adding to db...")

    for root, dirs, files in os.walk(sources_directory):
        for file in files:
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(root, file), encoding="utf8")
            elif file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root, file))
            elif file.endswith(".csv"):
                loader = CSVLoader(os.path.join(root, file))
            elif file.endswith(".epub"):
                loader = UnstructuredEPubLoader(os.path.join(root, file))
            elif file.endswith(".html"):
                loader = UnstructuredHTMLLoader(os.path.join(root, file))

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    qdrant = Qdrant.from_documents(texts, llama, path=db_dir, collection_name="test")
    qdrant = None
    print("Indexed ", len(texts), " documents in Qdrant")


if __name__ == "__main__":
    sources_directory = sys.argv[1] if len(sys.argv) > 1 else documents_directory
    cleandb = sys.argv[2] if len(sys.argv) > 2 else "n"
    main(sources_directory, cleandb)
