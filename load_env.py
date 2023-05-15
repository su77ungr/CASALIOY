"""load env variables"""
import os

from dotenv import load_dotenv

load_dotenv()

# generic
llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
model_n_ctx = int(os.environ.get("MODEL_N_CTX"))
use_mlock = os.environ.get("USE_MLOCK").lower() == "true"

# ingest
persist_directory = os.environ.get("PERSIST_DIRECTORY")
documents_directory = os.environ.get("DOCUMENTS_DIRECTORY")
chunk_size = int(os.environ.get("INGEST_CHUNK_SIZE"))
chunk_overlap = int(os.environ.get("INGEST_CHUNK_OVERLAP"))

# generate
model_type = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_temp = float(os.environ.get("MODEL_TEMP", "0.8"))
model_stop = os.environ.get("MODEL_STOP", "")
model_stop = model_stop.split(",") if model_stop else []
chain_type = os.environ.get("CHAIN_TYPE", "refine")
