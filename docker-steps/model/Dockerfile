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

RUN python3.9 -m pip install torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt ./

RUN sed -i '/torch==2.1.0+cu118/d' requirements.txt && \
    sed -i '/torchaudio==2.1.0+cu118/d' requirements.txt && \
    sed -i '/torchvision==0.16.0+cu118/d' requirements.txt

RUN python3.9 -m pip install -r requirements.txt --ignore-installed

FROM scratch as fs

COPY --from=antoniogrv/step-dataset-generation-config:latest transformers/ /transformers
COPY . .

FROM cuda as core

COPY --from=fs . .
COPY deps/transcript_label.pkl data/inputs_model/transcript_label.pkl

ENTRYPOINT python3.9 entrypoint.py