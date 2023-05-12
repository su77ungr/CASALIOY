import shutil
import os
import sys
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import LlamaCppEmbeddings

## enables to run python random_path/ to ingest // or 'python random_path/ y' to purge existing db 
def main(sources_directory, cleandb):
    db_dir = "./db"
    if os.path.exists(db_dir):
        
        if cleandb.lower() == 'y' or (cleandb == 'n' and input("\nDelete current database?(Y/N): ").lower() == 'y'):
            print('Deleting db...')
            shutil.rmtree(db_dir)
        elif cleandb.lower() == 'n':
            print('Adding to db...')

    for root, dirs, files in os.walk(sources_directory):
        for file in files:
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(root, file), encoding="utf8")
            elif file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root, file))
            elif file.endswith(".csv"):
                loader = CSVLoader(os.path.join(root, file))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    llama = LlamaCppEmbeddings(model_path="./models/ggml-model-q4_0.bin")
    qdrant = Qdrant.from_documents(texts, llama, path="./db", collection_name="test")
    qdrant = None
    print("Indexed ", len(texts), " documents in Qdrant")

if __name__ == "__main__":
    sources_directory = sys.argv[1]
    cleandb = sys.argv[2] if len(sys.argv) > 2 else 'n'
    main(sources_directory, cleandb)
