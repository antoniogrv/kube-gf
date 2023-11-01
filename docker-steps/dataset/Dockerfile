# Produce un'immagine Docker parametrica che genera i dataset da dare in pasto al modello.
# Rispettivamente, è possibile generare i dataset "train", "test" e "val" mediante gli NPM scripts nel package.json.

# Questo Dockerfile prevede un argomento "dataset_type_arg" che può assumere i valori sopraindicati; inoltre ..
# .. adotta la strategia delle multi-stage builds per snellire l'immagine finale.

# Per maggiori dettagli sui singoli stage, si rimanda alla documentazione ufficiale del progetto.

FROM python:3.8-slim-buster as pip_dependencies

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install genometools -y --no-install-recommends

FROM python:3.8-slim-buster as dnabert

RUN apt-get update && \
    apt-get install git -y --no-install-recommends

RUN git clone https://github.com/jerryji1993/DNABERT && \
    cd DNABERT && \
    python3 -m pip install Cmake . && \
    python3 -m pip install --editable . && \
    cd examples && \
    cd ../.. && \
    mv DNABERT/src/transformers ./transformers && \
    rm -r DNABERT

FROM pip_dependencies as transcripts

COPY data/genes_panel.txt data/download_transcripts.py ./

RUN python download_transcripts.py

FROM pip_dependencies as core

COPY . .
COPY --from=dnabert transformers/ /transformers
COPY --from=transcripts transcripts/ /data/transcripts/

ARG dataset_type_arg
ENV dataset_type_env=$dataset_type_arg

RUN echo "Preparazione del dataset di $dataset_type_env"
ENTRYPOINT python entrypoint.py -type $dataset_type_env 