import shutil
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import LlamaCppEmbeddings
from sys import argv
import os

def main():
    # Check if there is an existing db, and if so delete or add to it.
    # supports command line "python ./ingest.py y" or "python ./ingest.py n"
    # y=new db, n=existing db
    if os.path.exists("./db"):
        if len(argv[1]) > 0:
            cleandb = argv[1]
        else:
            cleandb = input("\nDelete current database?(Y/N): ")
        if cleandb.lower() == 'y':
            print('Deleting db...')
            shutil.rmtree("./db")
        elif cleandb.lower() == 'n':
            print('Adding to existing db...')
        else:
            print('No option selected!')
    # Load document and split in chunks
    for root, dirs, files in os.walk("source_documents"):
        for file in files:
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(root, file), encoding="utf8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    # Create embeddings
    llama = LlamaCppEmbeddings(model_path="./models/ggml-model-q4_0.bin")


    qdrant = Qdrant.from_documents(
    texts, llama, path="./db",  # Local mode with in-memory storage only
    collection_name="test",
)
    qdrant
    qdrant = None

if __name__ == "__main__":
    main()