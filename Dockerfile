FROM python:3.10

WORKDIR /srv

RUN git clone https://github.com/su77ungr/CASALIOY.git
COPY ./requirements.txt /srv/CASALIOY/

RUN python3 -m venv venv && . venv/bin/activate \
    && pip3 install -r /srv/CASALIOY/requirements.txt --upgrade pip

COPY ./ingest.py /srv/ingest.py
COPY ./startLLM.py /srv/startLLM.py
COPY ./example.env /srv/.env
