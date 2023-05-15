FROM python:3.11

WORKDIR /srv
RUN git clone https://github.com/su77ungr/CASALIOY.git
WORKDIR CASALIOY

RUN pip3 install poetry
RUN python3 -m poetry config virtualenvs.create false
RUN python3 -m poetry install
RUN python3 -m pip install --force streamlit sentence_transformers # Temp fix, see pyproject.toml
RUN python3 -m pip uninstall -y llama-cpp-python
RUN CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 python3 -m pip install llama-cpp-python  # GPU support
RUN pre-commit install
COPY example.env .env
