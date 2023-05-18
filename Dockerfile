###############################################
# Base Image
###############################################
FROM python:3.11-slim as python-base
# We set POETRY_VERSION=1.3.2 because 1.4.x has some weird legacy issues
# CASALIOY_FORCE_CPU = we install cpu-only pytorch.
ENV PYTHONFAULTHANDLER=1 \
      PYTHONUNBUFFERED=1 \
      PYTHONHASHSEED=random \
      PIP_NO_CACHE_DIR=off \
      PIP_DISABLE_PIP_VERSION_CHECK=on \
      PIP_DEFAULT_TIMEOUT=100 \
      POETRY_NO_INTERACTION=1 \
      POETRY_VIRTUALENVS_IN_PROJECT=true \
      POETRY_VERSION=1.3.2 \
      CASALIOY_FORCE_CPU=true
RUN apt-get update && apt-get install -y build-essential git htop gdb nano unzip curl && rm -rf /var/lib/apt/lists/*
#RUN if [ "$CASALIOY_ENABLE_LLAMA_GPU" = "true" ]; then \
#        apt-get install -y nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc; \
#    fi; \
RUN pip install --upgrade setuptools virtualenv

###############################################
# Builder Image
###############################################
FROM python-base as builder-base
RUN pip install "poetry==$POETRY_VERSION"
WORKDIR /srv
RUN git clone https://github.com/su77ungr/CASALIOY.git
WORKDIR CASALIOY
RUN poetry install --with GUI,LLM --without dev --sync
RUN . .venv/bin/activate && pip install --force streamlit
RUN . .venv/bin/activate && \
    if [ "$CASALIOY_FORCE_CPU" = "true" ]; then \
        pip install --force torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --force sentence_transformers; \
    fi

###############################################
# Production Image
###############################################
FROM python-base as production
COPY --from=builder-base /srv /srv
WORKDIR /srv/CASALIOY
COPY example.env .env
RUN echo "source /srv/CASALIOY/.venv/bin/activate" >> ~/.bashrc
RUN . .venv/bin/activate && python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
