"""load env variables"""
import os
from typing import Callable

from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from prompt_toolkit import HTML, PromptSession, print_formatted_text
from prompt_toolkit.styles import Style
from pyexpat import ExpatError

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
n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", 0))


def get_embedding_model() -> tuple[HuggingFaceEmbeddings | LlamaCppEmbeddings, Callable]:
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


def get_prompt_template_kwargs() -> dict[str, PromptTemplate]:
    """get an improved prompt template"""
    match chain_type:
        case "stuff":
            question_prompt = """HUMAN: Answer the question using ONLY the given context. If you are unsure of the answer, respond with "Unknown[STOP]". Conclude your response with "[STOP]" to indicate the completion of the answer.

Context: {context}

Question: {question}

ASSISTANT:"""
            return {"prompt": PromptTemplate(template=question_prompt, input_variables=["context", "question"])}
        case "refine":
            question_prompt = """HUMAN: Answer the question using ONLY the given context.
Indicate the end of your answer with "[STOP]" and refrain from adding any additional information beyond that which is provided in the context.

Question: {question}

Context: {context_str}

ASSISTANT:"""
            refine_prompt = """HUMAN: Refine the original answer to the question using the new context.
Use ONLY the information from the context and your previous answer.
If the context is not helpful, use the original answer.
Indicate the end of your answer with "[STOP]" and avoid adding any extraneous information.

Original question: {question}

Existing answer: {existing_answer}

New context: {context_str}

ASSISTANT:"""
            return {
                "question_prompt": PromptTemplate(template=question_prompt, input_variables=["context_str", "question"]),
                "refine_prompt": PromptTemplate(template=refine_prompt, input_variables=["context_str", "existing_answer", "question"]),
            }
        case _:
            return {}


style = Style.from_dict(
    {
        "r": "italic gray",  # remark
        "w": "italic yellow",  # warning
        "d": "bold red",  # danger
        "b": "bold",
        "i": "italic",
        "question": "ansicyan",
        "answer": "ansigreen",
        "source": "ansimagenta",
    }
)


def print_HTML(text: str, **kwargs) -> None:
    """print formatted HTML text"""
    try:
        print_formatted_text(HTML(text).format(**kwargs), style=style)
    except ExpatError:
        print(text)


def prompt_HTML(session: PromptSession, prompt: str, **kwargs) -> str:
    """print formatted HTML text"""
    try:
        return session.prompt(HTML(prompt).format(**kwargs), style=style)
    except ExpatError:
        return input(prompt)
