<div align="center">
<p align="center">


*****
  #  CASALIOY - Your local langchain toolkit 
  
  </p>
</div>

  <h2 align="center">

    
<p align="center">
    <img height="300" src="https://github.com/su77ungr/GEEB-GPT/assets/69374354/2e59734c-0de7-4057-be7a-14729e1d5acd" alt="Qdrant"></img><br>

  <a href="https://github.com/su77ungr/CASALIOY/issues/8"><img src="https://img.shields.io/badge/Feature-Requests-bc1439.svg" alt="Roadmap 2023"></a>
  
  <br> <br>


</p>
The fastest toolkit for air-gapped LLMs
 
[LangChain](https://github.com/hwchase17/langchain) + [LlamaCpp](https://pypi.org/project/llama-cpp-python/) + [qdrant](https://qdrant.tech/) (refers to slower [imartinez](https://github.com/imartinez/privateGPT)) 👀


# Setup your environment 

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

  
Then, download the 2 models and place them in a folder called `./models`:

- LLM: default is [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin) / run `/Demos/customLLM.py` (check paths) instead of `startLLM.py`
- Embedding: default to [ggml-model-q4_0.bin](https://huggingface.co/Pi3141/alpaca-native-7B-ggml/resolve/397e872bf4c83f4c642317a5bf65ce84a105786e/ggml-model-q4_0.bin). / Custom  Embeddings model, reference it in `/Demos/customLLM.py` and `ingest.py`.
 
This should look like this 
  
```
└── repo
      ├── startLLM.py
      ├── ingest.py
      ├── source_documents
      │   └── dsgvo.txt
      ├── models
      │   ├── ggml-gpt4all-j-v1.3-groovy.bin
      │   └── ggml-model-q4_0.bin
      └── Demos/
```
  
  
## Test dataset
This repo uses a [state of the union transcript](https://github.com/imartinez/privateGPT/blob/main/source_documents/state_of_the_union.txt) as an example.

## Ingesting your own dataset

Get your .txt files ready inside your ``` <path_to_your_data_directory> ```. 

To ingest the data run (auto-ingest .txt, .pdf, .csv)

```shell
python ingest.py  <path_to_your_data_directory>
```
Optional: use `y` flag to purge existing vectorstore and initialize fresh instance
```shell
python ingest.py <path_to_your_data_directory> y 
```

This spins up a local qdrant namespace inside the `db` folder containing the local vectorstore. Will take time, depending on the size of your document.
You can ingest as many documents as you want by running `ingest`, and all will be accumulated in the local embeddings database. To remove dataset simply remove `db` folder. 

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python startLLM.py
```

And wait for the script to require your input. 

```shell
> Enter a query:
```

Hit enter. You'll need to wait 20-30 seconds (depending on your machine) while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again. 

Note: you could turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.
  
  
# LLM options

  ### Optional / Custom models outside of the GPT-J ecosphere  (NEW) 
  
| Model                     |  BoolQ   |   PIQA   | HellaSwag | WinoGrande |  ARC-e   |  ARC-c   |   OBQA   |   Avg.   |
|:--------------------------|:--------:|:--------:|:---------:|:----------:|:--------:|:--------:|:--------:|:--------:|
| [ggml-vic-7b-uncensored](https://cdn-lfs.huggingface.co/repos/d5/aa/d5aaf35e7d0d28440ac96a9c64b5c2a17e2e3fc260e1c41133376a6918b172a2/e682acb5b798df30cb06d7953a5e08956f73f4d480327ead19336d08a1658112?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27ggml-vic7b-uncensored-q4_0.bin%3B+filename%3D%22ggml-vic7b-uncensored-q4_0.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1684006500&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zL2Q1L2FhL2Q1YWFmMzVlN2QwZDI4NDQwYWM5NmE5YzY0YjVjMmExN2UyZTNmYzI2MGUxYzQxMTMzMzc2YTY5MThiMTcyYTIvZTY4MmFjYjViNzk4ZGYzMGNiMDZkNzk1M2E1ZTA4OTU2ZjczZjRkNDgwMzI3ZWFkMTkzMzZkMDhhMTY1ODExMj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODQwMDY1MDB9fX1dfQ__&Signature=qqcSgIbRTN6DNkKQaN-1Ihp4isuSq-HGjx6hfKDs6T0%7ERtgZwWcovKOdRV71ucjQR%7EdIe4ZC6aGguK9j9KhxNABhseRcYAMWfI-wNOg07eN8h0REgqu42ePdsy8T-%7E2FaWALoJtY3lVPcYs%7ES8xq8fWBN9aU-2Eam-lnkr%7ExjCr3n9GHTXxpa3abwDv1%7E4oqxHSjvwmGWuW7BxIyZlJOUhdTT4acbL1wzYDaOqq36hp2JK6MNssxwK1e0xbgP19NqbUWMaml7P1c%7ErYWgBpDsqRYJ3cPteEDAYcURlNuuQ-MBgEmG17WmppEJiS4uG9-VQ2C5YLvFj4ksK53NIwW9g__&Key-Pair-Id=KVTP0A1DKRTAX)         |   73.4   |   74.8   |   63.4    |    64.7    |   54.9   |   36.0   |   40.2   |   58.2    |
| [gpt4all-13b-snoozy q5](https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML/blob/main/GPT4All-13B-snoozy.ggml.q5_1.bin)        | 83.3 |   79.2   |   75.0    |  71.3  |   60.9   |   44.2   |   43.4   |  65.3  |
  
  
  ###  Optional / Custom models inside of the GPT-J ecosphere


  
  
| Model                     |  BoolQ   |   PIQA   | HellaSwag | WinoGrande |  ARC-e   |  ARC-c   |   OBQA   |   Avg.   |
|:--------------------------|:--------:|:--------:|:---------:|:----------:|:--------:|:--------:|:--------:|:--------:|
| GPT4All-J 6B v1.0         |   73.4   |   74.8   |   63.4    |    64.7    |   54.9   |   36.0   |   40.2   |   58.2   |
| [GPT4All-J v1.1-breezy](https://gpt4all.io/models/ggml-gpt4all-j-v1.1-breezy.bin)      |   74.0   |   75.1   |   63.2    |    63.6    |   55.4   |   34.9   |   38.4   |   57.8   |
| [GPT4All-J v1.2-jazzy](https://gpt4all.io/models/ggml-gpt4all-j-v1.2-jazzy.bin)       |   74.8   |   74.9   |   63.6    |    63.8    |   56.6   |   35.3   |   41.0   |   58.6   |
| [GPT4All-J v1.3-groovy](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin)     |   73.6   |   74.3   |   63.8    |    63.5    |   57.7   |   35.0   |   38.8   |   58.1   |
| [GPT4All-J Lora 6B](https://gpt4all.io/models/)         |   68.6   |   75.8   |   66.2    |    63.5    |   56.4   |   35.7   |   40.2   |   58.1   |
  
   all the supported models from [here](https://huggingface.co/nomic-ai/gpt4all-13b-snoozy) (custom LLMs in Pipeline)
  
  


## How does it work? 👀

<img src="https://qdrant.tech/articles_data/langchain-integration/flow-diagram.png"></img>

Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `LlamaCppEmbeddings`. It then stores the result in a local vector database using `Qdrant` vector store. 
  <br>
    <img height="100" src="https://github.com/qdrant/qdrant/raw/master/docs/logo.svg" alt="Qdrant">  
- `startLLM.py` can  handle every LLM that is llamacpp compatible (default `GPT4All-J`). The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
  
  <p align="center">

</p>
  
## Pipeline (stuff to do) 🧑‍🎤
  
  - ⭕ Adding better documentation
  
  - ⭕ Retrieval with Contextual Compression Retriever or custom Retrieval Algorithm
  
  - ⭕ Custom LLM endpoints via Hugging Face Pipelines [see](https://github.com/su77ungr/CASALIOY/issues/2)
  
  - [done] Custom LLM integration via native LlamaCpp see Demos/*

  - [done] Adding auto-parser for datatypes
  
  - ♾️ README.md updates



## 💁 Contributing

<div align="center">
<a href="https://github.com/su77ungr/CASALIOY/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=su77ungr/CASALIOY" />
</a><br><br>
  <a href="https://www.buymeacoffee.com/cassowary"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a pizza&emoji=🌶️&slug=cassowary&button_colour=f0e4ff&font_colour=000000&font_family=Bree&outline_colour=000000&coffee_colour=FFDD00" /></a>

</div>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=su77ungr/CASALIOY&type=Date)](https://star-history.com/#su77ungr/CASALIOY&Date)

  
# Disclaimer
The contents of this repository are provided "as is" and without warranties of any kind, whether express or implied. We do not warrant or represent that the information contained in this repository is accurate, complete, or up-to-date. We expressly disclaim any and all liability for any errors or omissions in the content of this repository.

Furthermore, this repository may contain links to other repositories or websites, which are not under our control. We do not endorse any of these repositories or websites and we are not responsible for their content or availability. We do not guarantee that any of the links provided on this repository will be free of viruses or other harmful components. We hereby exclude liability for any losses or damages that may arise from the use of any links on this or any linked repository or website.

In particular, we make no express or implied representations or warranties regarding the accuracy, completeness, suitability, reliability, availability, or timeliness of any information, products, services, or related graphics contained in this repository for any purpose. We hereby exclude all conditions, warranties, representations, or other terms which may apply to this repository or any content in it, whether express or implied.

We also hereby exclude any liability for any damages or losses arising from the use of binaries in this repository. You acknowledge and agree that any use of any binaries in this repository is at your own risk.

By using this repository, you are agreeing to comply with and be bound by the above disclaimer. If you do not agree with any part of this disclaimer, please do not use this repository.
