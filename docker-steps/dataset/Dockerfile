# Produce un'immagine Docker parametrica che genera i dataset da dare in pasto al modello.
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

FROM ubuntu as fusim

RUN apt-get update && \
    apt-get install wget unzip -y --no-install-recommends

RUN wget https://github.com/aebruno/fusim/raw/master/releases/fusim-0.2.2-bin.zip --no-check-certificate && \
    unzip fusim-0.2.2-bin.zip && \
    rm fusim-0.2.2-bin.zip

RUN wget -O refFlat.txt.gz http://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/refFlat.txt && \
    gunzip refFlat.txt.gz && \
    mv refFlat.txt fusim-0.2.2/refFlat.txt

RUN wget ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz && \
    tar -xzf chromFa.tar.gz && \
    cat chr*.fa > fusim-0.2.2/hg19.fa

FROM pip_dependencies as core

COPY --from=fusim fusim-0.2.2/scripts /fusim-0.2.2/scripts
COPY --from=fusim fusim-0.2.2/fusim.jar /fusim-0.2.2/fusim.jar
COPY --from=fusim fusim-0.2.2/hg19.fa /fusim-0.2.2/hg19.fa
COPY --from=fusim fusim-0.2.2/refFlat.txt /fusim-0.2.2/refFlat.txt

COPY --from=dnabert transformers/ /transformers
COPY --from=transcripts transcripts/ /data/transcripts/

RUN apt-get update && \
    apt-get install samtools art-nextgen-simulation-tools -y --no-install-recommends && \
    samtools faidx fusim-0.2.2/hg19.fa

RUN apt-get install default-jre -y --no-install-recommends

COPY . .

ENTRYPOINT python gc_handler.py