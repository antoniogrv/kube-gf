#

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

CMD python entrypoint.py