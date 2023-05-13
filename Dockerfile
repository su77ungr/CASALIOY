FROM ubuntu:latest

RUN apt-get update -qq && apt-get install -y \
    git \
    python3 \
    python-is-python3 \
    python3-pip \
    wget

RUN cd /home \
    && git clone https://github.com/su77ungr/CASALIOY.git \
    && cd CASALIOY/ \
    && pip3 install -r requirements.txt

RUN rule='\n,\t' && echo -e "PERSIST_DIRECTORY=db\nDOCUMENTS_DIRECTORY=source_documents\nMODEL_TYPE=LlamaCpp\nLLAMA_EMBEDDINGS_MODEL=models/ggml-model-q4_0.bin\nMODEL_PATH=models/ggjt-v1-vic7b-uncensored-q4_0.bin\nMODEL_N_CTX=512\nMODEL_TEMP=0.8\nMODEL_STOP=${rule}" > /home/CASALIOY/.env \
    && chmod a+x /home/CASALIOY/.env

RUN cd /home/CASALIOY/models \
    && wget https://huggingface.co/datasets/dnato/ggjt-v1-vic7b-uncensored-q4_0.bin/resolve/main/ggjt-v1-vic7b-uncensored-q4_0.bin \
    && wget https://huggingface.co/Pi3141/alpaca-native-7B-ggml/resolve/397e872bf4c83f4c642317a5bf65ce84a105786e/ggml-model-q4_0.bin
