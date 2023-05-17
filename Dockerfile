###############################################
# Base Image
###############################################
FROM python:3.11-slim as python-base
ENV PYTHONFAULTHANDLER=1 \
      PYTHONUNBUFFERED=1 \
      PYTHONHASHSEED=random \
      PIP_NO_CACHE_DIR=off \
      PIP_DISABLE_PIP_VERSION_CHECK=on \
      PIP_DEFAULT_TIMEOUT=100 \
      POETRY_NO_INTERACTION=1 \
      POETRY_VIRTUALENVS_IN_PROJECT=true \
      PATH="$PATH:/srv/CASALIOY/.venv/bin" \
      PYTHONPATH="$PYTHONPATH:/srv/CASALIOY/.venv/lib/python3.11/site-packages" \
      POETRY_VERSION=1.4.2

###############################################
# Builder Image
###############################################
FROM python-base as builder-base
# System deps:
RUN apt-get update && apt-get install -y build-essential git
RUN pip install "poetry==$POETRY_VERSION"
WORKDIR /srv
RUN git clone https://github.com/hippalectryon-0/CASALIOY.git
WORKDIR CASALIOY
RUN git checkout better-docker
RUN pip install --upgrade setuptools virtualenv
RUN poetry install --with GUI,LLM --without dev --sync
RUN . .venv/bin/activate && pip install --force torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU-only torch for sentence_transformers
RUN . .venv/bin/activate && pip install --force streamlit # Temp fix, see pyproject.toml

###############################################
# Production Image
###############################################
FROM python-base as production
COPY --from=builder-base /srv /srv
WORKDIR /srv/CASALIOY
COPY example.env .env
