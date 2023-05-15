"""answer questions using documents from LibGen"""
import asyncio
import os
import shutil
from pathlib import Path

from libgenesis import Libgen
from prompt_toolkit import prompt

from casalioy.ingest import Ingester
from casalioy.load_env import (
    chunk_overlap,
    chunk_size,
    get_embedding_model,
    model_n_ctx,
    model_path,
    model_stop,
    model_temp,
    n_gpu_layers,
    persist_directory,
    use_mlock,
)
from casalioy.startLLM import QASystem

max_doc_size_mb = 10
out_path = Path("source_documents/libgen")

if out_path.exists():
    shutil.rmtree(out_path)
os.mkdir(out_path)


def load_documents(keyword: str, n: int = 3) -> None:
    """load random documents from LG using keyword"""
    lg = Libgen()
    result = asyncio.run(lg.search(keyword))
    dl_N = 0
    for item_id in result:
        if dl_N >= n:
            break
        item = result[item_id]
        if int(item["filesize"]) > 1024**2 * max_doc_size_mb:
            continue
        if item["extension"] not in ["pdf", "epub"]:
            print(f"skipped ext. {item['extension']}")
            continue
        asyncio.run(lg.download(item["mirrors"]["main"], dest_folder=out_path))
        dl_N += 1
    if dl_N == 0:
        raise ValueError(f"No good result for {keyword}")
    print(f"Got {dl_N} files")


def search(question: str, keyword: str) -> None:
    """ask a question"""
    load_documents(keyword)

    Ingester(persist_directory, collection="libgen").ingest_from_directory(str(out_path), chunk_size, chunk_overlap)

    qa = QASystem(get_embedding_model()[0], persist_directory, model_path, model_n_ctx, model_temp, model_stop, use_mlock, n_gpu_layers, collection="libgen")
    qa.prompt_once(question)


if __name__ == "__main__":
    question = prompt("Enter your question: ")
    keyword = prompt("Enter a keyword to search for relevant sources: ")

    search(question, keyword)
