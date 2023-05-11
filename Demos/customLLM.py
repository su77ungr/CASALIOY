from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Qdrant
import qdrant_client
from langchain.llms import LlamaCpp



def main():
    # Load stored vectorstore
    llama = LlamaCppEmbeddings(model_path='../models/ggml-model-q4_0.bin')
    # Load ggml-formatted model 
    local_path = '../models/ggml-vic7b-uncensored-q4_0.bin'

    client = qdrant_client.QdrantClient(
    path="./db", prefer_grpc=True
    )
    qdrant = Qdrant(
        client=client, collection_name="test", 
        embeddings=llama
    )

    # Prepare the LLM chain 
    callbacks = [StreamingStdOutCallbackHandler()]
    #llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True, backend='gptj')
    llm = LlamaCpp(
    model_path=local_path, callbacks=callbacks, verbose=True)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=qdrant.as_retriever(search_type="mmr"), return_source_documents=True)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

if __name__ == "__main__":
    main()
