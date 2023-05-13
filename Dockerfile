FROM python:3.10

WORKDIR /srv
COPY ./requirements.txt .

RUN python3 -m venv venv && . venv/bin/activate
RUN python3 -m pip install --no-cache-dir -r requirements.txt --upgrade pip

COPY ./ingest.py /srv/ingest.py
COPY ./startLLM.py /srv/startLLM.py
COPY ./example.env /srv/.env

# COPY ./models /srv/models  # Mounting model is more efficient
