"""load env variables"""
import os
from typing import Callable

from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings

load_dotenv()

# generic
text_embeddings_model = os.environ.get("TEXT_EMBEDDINGS_MODEL")
text_embeddings_model_type = os.environ.get("TEXT_EMBEDDINGS_MODEL_TYPE")
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


def get_embedding_model() -> tuple[HuggingFaceEmbeddings, Callable] | tuple[LlamaCppEmbeddings, Callable]:
    """get the text embedding model
    :returns: tuple[the model, its encoding function]"""
    match text_embeddings_model_type:
        case "HF":
            model = HuggingFaceEmbeddings(model_name=text_embeddings_model)
            return model, model.client.encode
        case "LlamaCpp":
            model = LlamaCppEmbeddings(model_path=text_embeddings_model, n_ctx=model_n_ctx)
            return model, model.client.embed
        case _:
            raise ValueError(f"Unknown embedding type {text_embeddings_model_type}")
