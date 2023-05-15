"""start the local LLM"""

import qdrant_client
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Qdrant
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text.html import html_escape

from casalioy.load_env import (
    chain_type,
    get_embedding_model,
    get_prompt_template_kwargs,
    model_n_ctx,
    model_path,
    model_stop,
    model_temp,
    model_type,
    n_gpu_layers,
    persist_directory,
    print_HTML,
    prompt_HTML,
    use_mlock,
)


class QASystem:
    """custom QA system"""

    def __init__(
        self,
        embeddings: Embeddings,
        db_path: str,
        model_path: str,
        n_ctx: int,
        temperature: float,
        stop: list[str],
        use_mlock: bool,
        n_gpu_layers: int,
        collection="test",
    ):
        # Get embeddings and local vector store
        client = qdrant_client.QdrantClient(path=db_path, prefer_grpc=True)
        qdrant = Qdrant(client=client, collection_name=collection, embeddings=embeddings)

        # Prepare the LLM chain
        callbacks = [StreamingStdOutCallbackHandler()]
        match model_type:
            case "LlamaCpp":
                from langchain.llms import LlamaCpp

                llm = LlamaCpp(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    temperature=temperature,
                    stop=stop,
                    callbacks=callbacks,
                    verbose=True,
                    n_threads=6,
                    n_batch=1000,
                    use_mlock=use_mlock,
                )
                # Need this hack because this param isn't yet supported by the python lib
                state = llm.client.__getstate__()
                state["n_gpu_layers"] = n_gpu_layers
                llm.client.__setstate__(state)
            case "GPT4All":
                from langchain.llms import GPT4All

                llm = GPT4All(
                    model=model_path,
                    n_ctx=n_ctx,
                    callbacks=callbacks,
                    verbose=True,
                    backend="gptj",
                )
            case _:
                raise ValueError("Only LlamaCpp or GPT4All supported right now. Make sure you set up your .env correctly.")

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=qdrant.as_retriever(search_type="mmr"),
            return_source_documents=True,
            chain_type_kwargs=get_prompt_template_kwargs(),
        )

    def prompt_once(self, query: str) -> None:
        """run a prompt"""
        # Get the answer from the chain
        res = self.qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        sources_str = "\n\n".join(f">> <source>{html_escape(document.metadata['source'])}</source>:\n{html_escape(document.page_content)}" for document in docs)
        print_HTML(
            f"\n\n> <question><b>Question</b>: {query}</question>\n> <answer><b>Answer</b>: {answer}</answer>\n> <b>Sources</b>:\n{sources_str}",
            query=query,
            answer=answer,
            sources_str=sources_str,
        )


# noinspection PyMissingOrEmptyDocstring
def main() -> None:
    session = PromptSession(auto_suggest=AutoSuggestFromHistory())
    qa_system = QASystem(get_embedding_model()[0], persist_directory, model_path, model_n_ctx, model_temp, model_stop, use_mlock, n_gpu_layers)
    while True:
        query = prompt_HTML(session, "\n<b>Enter a query</b>: ").strip()
        if query == "exit":
            break
        elif not query:  # check if query empty
            print_HTML("<r>Empty query, skipping</r>")
            continue
        qa_system.prompt_once(query)


if __name__ == "__main__":
    main()
