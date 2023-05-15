"""start the local LLM"""

import qdrant_client
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant

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
    use_mlock,
)


def initialize_qa_system() -> RetrievalQA:
    """init the LLM"""
    # Get embeddings and local vector store
    embeddings = get_embedding_model()[0]
    client = qdrant_client.QdrantClient(path=persist_directory, prefer_grpc=True)
    qdrant = Qdrant(client=client, collection_name="test", embeddings=embeddings)

    # Prepare the LLM chain
    callbacks = [StreamingStdOutCallbackHandler()]
    match model_type:
        case "LlamaCpp":
            from langchain.llms import LlamaCpp

            llm = LlamaCpp(
                model_path=model_path,
                n_ctx=model_n_ctx,
                temperature=model_temp,
                stop=model_stop,
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
                n_ctx=model_n_ctx,
                callbacks=callbacks,
                verbose=True,
                backend="gptj",
            )
        case _:
            raise ValueError("Only LlamaCpp or GPT4All supported right now. Make sure you set up your .env correctly.")

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=qdrant.as_retriever(search_type="mmr"),
        return_source_documents=True,
        chain_type_kwargs=get_prompt_template_kwargs(),
    )


# noinspection PyMissingOrEmptyDocstring
def main() -> None:
    qa_system = initialize_qa_system()
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ").strip()
        if query == "exit":
            break
        elif not query:  # check if query empty
            print("Empty query, skipping")
            continue

        # Get the answer from the chain
        res = qa_system(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        sources_str = "\n\n".join(f">> {document.metadata['source']}:\n{document.page_content}" for document in docs)
        print(
            f"""\n\n> Question: {query}
> Answer: {answer}
> Sources:\n{sources_str}"""
        )


if __name__ == "__main__":
    main()
