<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
      <h4>
      Helpful? ⚡ bring me closer to
            <a href="https://www.evwest.com/catalog/product_info.php?cPath=40&products_id=168">convert</a>
            my car into electric <a href="https://www.buymeacoffee.com/cassowary" target="_blank"><img src="https://github.com/su77ungr/CASALIOY/assets/69374354/1de2e6a1-d4ee-44a8-baf1-db4f2765b833" alt="Buy Me A Coffee" height="30" width="30"></a>  
      </h4>
      
#   
      
<h2>
<p>
<img height="400" src="https://github.com/su77ungr/CASALIOY/assets/69374354/41c1edcb-af2f-4327-bdcd-5248ffe6ecf9" alt="Qdrant"><br>
      
<a href="https://github.com/su77ungr/CASALIOY/issues/8"><img src="https://img.shields.io/badge/Feature-Requests-bc1439.svg" alt="Roadmap 2023"> [![Docker Pulls](https://badgen.net/docker/pulls/su77ungr/casalioy?icon=docker&label=pulls)](https://hub.docker.com/r/su77ungr/casalioy/)</a>
![example workflow](https://github.com/su77ungr/CASALIOY/actions/workflows/docker-image.yml/badge.svg)
<br>
</p>
      
The fastest toolkit for air-gapped LLMs with <a  href="#chat-inside-gui-new-feature"><img src="https://img.shields.io/badge/GUI-blue.svg" alt="Roadmap 2023">
      
[LangChain](https://github.com/hwchase17/langchain) + [LlamaCpp](https://pypi.org/project/llama-cpp-python/) + [qdrant](https://qdrant.tech/)
      
</div>
         
<br>
     
      
# Setup

### Docker ( 🚰 under construction. tested on Ubuntu LTS) 

```bash
docker pull su77ungr/casalioy:stable
```

```bash
docker run -it -p 8501:8501 --shm-size=16gb su77ungr/casalioy:stable /bin/bash
```
for GPU support of stable use `casalioy:gpu` (unstable)

> All set! Proceed with ingesting your [dataset](#ingesting-your-own-dataset)

### Build it from source

> First install all requirements:

```shell
python -m pip install poetry
python -m poetry config virtualenvs.in-project true
python -m poetry install
. .venv/bin/activate
python -m pip install --force streamlit sentence_transformers  # Temporary bandaid fix, waiting for streamlit >=1.23
pre-commit install
```

If you want GPU support for llama-ccp:

```shell
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --force llama-cpp-python
```

> Edit the example.env to fit your models and rename it to .env

```env
# Generic
MODEL_N_CTX=1024
TEXT_EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
TEXT_EMBEDDINGS_MODEL_TYPE=HF  # LlamaCpp or HF
USE_MLOCK=true

# Ingestion
PERSIST_DIRECTORY=db
DOCUMENTS_DIRECTORY=source_documents
INGEST_CHUNK_SIZE=500
INGEST_CHUNK_OVERLAP=50

# Generation
MODEL_TYPE=LlamaCpp # GPT4All or LlamaCpp
MODEL_PATH=eachadea/ggml-vicuna-7b-1.1/ggml-vic7b-q5_1.bin
MODEL_TEMP=0.8
MODEL_STOP=[STOP]
CHAIN_TYPE=stuff
N_RETRIEVE_DOCUMENTS=100 # How many documents to retrieve from the db
N_FORWARD_DOCUMENTS=6 # How many documents to forward to the LLM, chosen among those retrieved
```

This should look like this

```
└── repo
      ├── startLLM.py
      ├── casalioy
      │   └── ingest.py, load_env.py, startLLM.py, gui.py, ...
      │   └── misc/ 
      ├── source_documents
      │   └── sample.csv
      │   └── ...
      ├── models
      │   ├── ggml-vic7b-q5_1.bin
      │   └── ...
      └── .env, Dockerfile, ...
```


> 👇 Update your installation!


      git pull && poetry install



## Ingesting your own dataset

To automatically ingest different data types (.txt, .pdf, .csv, .epub, .html, .docx, .pptx, .eml, .msg)

> This repo includes dummy [files](https://github.com/su77ungr/CASALIOY/tree/main/source_documents)
> inside `source_documents` to run tests with.

```shell
python casalioy/ingest.py # optional <path_to_your_data_directory>
```

Optional: use `y` flag to purge existing vectorstore and initialize fresh instance

```shell
python casalioy/ingest.py # optional <path_to_your_data_directory> y
```

This spins up a local qdrant namespace inside the `db` folder containing the local vectorstore. Will take time,
depending on the size of your document.
You can ingest as many documents as you want by running `ingest`, and all will be accumulated in the local embeddings
database. To remove dataset simply remove `db` folder.

## Ask questions to your documents, locally!

In order to ask a question, run a command like:

```shell
python casalioy/startLLM.py
```

And wait for the script to require your input.

```shell
> Enter a query:
```

Hit enter. You'll need to wait 20-30 seconds (depending on your machine) while the LLM model consumes the prompt and
prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you
can then ask another question without re-running the script, just wait for the prompt again.

Note: you could turn off your internet connection, and the script inference would still work. No data gets out of your
local environment.

Type `exit` to finish the script.

## Chat inside GUI (new feature)

Introduced by [@alxspiker](https://github.com/alxspiker) -> see [#21](https://github.com/su77ungr/CASALIOY/pull/21)

```shell
streamlit run casalioy/gui.py
```

# LLM options       
   
### Leaderboard 
      
List of available open LLMs [HuggingFace](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
### models outside of the GPT-J ecosphere  (work out of the box)

> 🪢 avoid using non-v3 models when using other quantization than q5 (LlamaCpp introduced [v3](https://github.com/ggerganov/llama.cpp/pull/1508) for ggml)
      
| Model                                                                                                                                            | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg. |
|:-------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:----:|:---------:|:----------:|:-----:|:-----:|:----:|:----:|
| [GPT4All-13b-snoozy GGMLv3](https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML/blob/main/GPT4All-13B-snoozy.ggmlv3.q4_0.bin) 
  [GPT4All-13b-snoozy (deprecated)](https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin)                | 83.3  | 79.2 |   75.0    |    71.3    | 60.9  | 44.2  | 43.4 | 65.3 | 
      
      
      
### models inside of the GPT-J ecosphere

| Model                                                                             | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg. |
|:----------------------------------------------------------------------------------|:-----:|:----:|:---------:|:----------:|:-----:|:-----:|:----:|:----:|
| [GPT4All-J vanilla](https://gpt4all.io/models/ggml-gpt4all-j.bin)                                                                 | 73.4  | 74.8 |   63.4    |    64.7    | 54.9  | 36.0  | 40.2 | 58.2 |
| [GPT4All-J v1.1-breezy](https://gpt4all.io/models/ggml-gpt4all-j-v1.1-breezy.bin) | 74.0  | 75.1 |   63.2    |    63.6    | 55.4  | 34.9  | 38.4 | 57.8 |
| [GPT4All-J v1.2-jazzy](https://gpt4all.io/models/ggml-gpt4all-j-v1.2-jazzy.bin)   | 74.8  | 74.9 |   63.6    |    63.8    | 56.6  | 35.3  | 41.0 | 58.6 |
| [GPT4All-J v1.3-groovy](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin) | 73.6  | 74.3 |   63.8    |    63.5    | 57.7  | 35.0  | 38.8 | 58.1 |
| [GPT4All-J Lora 6B](https://gpt4all.io/models/)                                   | 68.6  | 75.8 |   66.2    |    63.5    | 56.4  | 35.7  | 40.2 | 58.1 |

all the supported models from [here](https://huggingface.co/nomic-ai/gpt4all-13b-snoozy) (custom LLMs in Pipeline)

### Convert GGML model to GGJT-ready model v1 (for truncation error or not supported models)

1. Download ready-to-use models

> Browse Hugging Face for [models](https://huggingface.co/)

2. Convert locally

> ``` python casalioy/misc/convert.py``` [see discussion](https://github.com/su77ungr/CASALIOY/issues/10#issue-1706854398)

# Pipeline

<br><br>

<img src="https://qdrant.tech/articles_data/langchain-integration/flow-diagram.png"></img>
<br><br>

Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data
leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `LlamaCppEmbeddings`. It
  then stores the result in a local vector database using `Qdrant` vector store.

- `startLLM.py` can handle every LLM that is llamacpp compatible (default `GPT4All-J`). The context for the answers is
  extracted from the local vector store using a similarity search to locate the right piece of context from the docs.

<br><br>


# Disclaimer

The contents of this repository are provided "as is" and without warranties of any kind, whether express or implied. We
do not warrant or represent that the information contained in this repository is accurate, complete, or up-to-date. We
expressly disclaim any and all liability for any errors or omissions in the content of this repository.

By using this repository, you are agreeing to comply with and be bound by the above disclaimer. If you do not agree with
any part of this disclaimer, please do not use this repository.
