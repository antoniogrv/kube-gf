# Produce un'immagine Docker dei modelli monolitici originali GeneFusion.

# Fusim Tools
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

# DNA-BERT Transformer
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

# Main Stage - CUDA
FROM nvidia/cuda:12.2.2-base-ubuntu22.04 as cuda

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        git \
        curl

RUN apt-get install -y \ 
        python3.9 \
        python3.9-distutils \
        libglib2.0-0

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    python3.9 -m pip install --upgrade pip

RUN apt-get update && \
    apt-get install genometools samtools art-nextgen-simulation-tools -y --no-install-recommends

RUN apt-get install default-jre -y --no-install-recommends

RUN python3.9 -m pip install torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu118

RUN git clone https://github.com/antoniogrv/kubeless-gf.git

WORKDIR /kubeless-gf

RUN sed -i '/torch==2.1.0+cu118/d' requirements.txt && \
    sed -i '/torchaudio==2.1.0+cu118/d' requirements.txt && \
    sed -i '/torchvision==0.16.0+cu118/d' requirements.txt

RUN python3.9 -m pip install -r requirements.txt --ignore-installed

RUN python3.9 data/download_transcripts.py

COPY --from=fusim fusim-0.2.2/scripts fusim-0.2.2/scripts
COPY --from=fusim fusim-0.2.2/refFlat.txt fusim-0.2.2/refFlat.txt
COPY --from=fusim fusim-0.2.2/fusim.jar fusim-0.2.2/fusim.jar
COPY --from=fusim fusim-0.2.2/hg19.fa fusim-0.2.2/hg19.fa

COPY --from=dnabert transformers/ transformers

RUN samtools faidx fusim-0.2.2/hg19.fa

ENTRYPOINT ["python3.9", "train_gene_classifier.py"]