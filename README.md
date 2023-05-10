## CASALIOY - GPT

  <h2 align="center">

    
    
<p align="center">
  
  <img height="300" src="https://github.com/su77ungr/GEEB-GPT/assets/69374354/2e59734c-0de7-4057-be7a-14729e1d5acd" alt="Qdrant">   

  <a href="https://git.io/su77ungr"><img src="https://img.shields.io/badge/Roadmap-2024-bc1439.svg" alt="Roadmap 2023"></a>
  
  <br> <br>

![run](https://github.com/su77ungr/CASALIOY/assets/69374354/9977296f-26fe-4841-ab95-be72a31774e8)

</p>

       Air-gapped LLMs on consumer-grade hardware 


<p align="center">


Built with [LangChain](https://github.com/hwchase17/langchain) and [GPT4All](https://github.com/nomic-ai/gpt4all) and hard-forked from [Imartinez](https://github.com/imartinez/privateGPT) 👀



# Environment Setup

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

Then, download the 2 models and place them in a folder called `./models`:
- LLM: default to [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin). If you prefer a different GPT4All-J compatible model, just download it and reference it in `privateGPT.py`.
- Embedding: default to [ggml-model-q4_0.bin](https://huggingface.co/Pi3141/alpaca-native-7B-ggml/resolve/397e872bf4c83f4c642317a5bf65ce84a105786e/ggml-model-q4_0.bin). If you prefer a different compatible Embeddings model, just download it and reference it in `privateGPT.py` and `ingest.py`.

## Test dataset
This repo uses a [state of the union transcript](https://github.com/imartinez/privateGPT/blob/main/source_documents/state_of_the_union.txt) as an example.

## Instructions for ingesting your own dataset

Get your .txt file ready.

Run the following command to ingest the data.

```shell
python ingest.py <path_to_your_txt_file>
```

It will create a `db` folder containing the local vectorstore. Will take time, depending on the size of your document.
You can ingest as many documents as you want by running `ingest`, and all will be accumulated in the local embeddings database. 
If you want to start from scracth, delete the `db` folder.

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python privateGPT.py
```

And wait for the script to require your input. 

```shell
> Enter a query:
```

Hit enter. You'll need to wait 20-30 seconds (depending on your machine) while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again. 

Note: you could turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

## How does it work? 👀
Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `LlamaCppEmbeddings`. It then stores the result in a local vector database using `Qdrant` vector store. 
  <br>
    <img height="100" src="https://github.com/qdrant/qdrant/raw/master/docs/logo.svg" alt="Qdrant">  
- `privateGPT.py` uses a local LLM based on `GPT4All-J` to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
- `GPT4All-J` wrapper was introduced in LangChain 0.0.162.

  <p align="center">

</p>
  
## Pipeline 🧑‍🎤
  
  - ⭕ Adding auto-parser for immutable types of data (i.e PDF, JSON, MD)
  
  - ⭕ Adding better documentation
  
  - ⭕ Adding support for faster and more secure Retrieval with Contextual Compression Retriever 
  
  - ♾️ README.md updates


 

## 💁 Contributing

As an open source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infra, or better documentation.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=su77ungr/CASALIOY&type=Date)](https://star-history.com/#su77ungr/CASALIOY&Date)

  
# Disclaimer
The contents of this repository are provided "as is" and without warranties of any kind, whether express or implied. We do not warrant or represent that the information contained in this repository is accurate, complete, or up-to-date. We expressly disclaim any and all liability for any errors or omissions in the content of this repository.

Furthermore, this repository may contain links to other repositories or websites, which are not under our control. We do not endorse any of these repositories or websites and we are not responsible for their content or availability. We do not guarantee that any of the links provided on this repository will be free of viruses or other harmful components. We hereby exclude liability for any losses or damages that may arise from the use of any links on this or any linked repository or website.

In particular, we make no express or implied representations or warranties regarding the accuracy, completeness, suitability, reliability, availability, or timeliness of any information, products, services, or related graphics contained in this repository for any purpose. We hereby exclude all conditions, warranties, representations, or other terms which may apply to this repository or any content in it, whether express or implied.

We also hereby exclude any liability for any damages or losses arising from the use of binaries in this repository. You acknowledge and agree that any use of any binaries in this repository is at your own risk.

By using this repository, you are agreeing to comply with and be bound by the above disclaimer. If you do not agree with any part of this disclaimer, please do not use this repository.
