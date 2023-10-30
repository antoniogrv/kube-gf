# kmer-kubeful step: dataset-config

# -- Stage 1 -- #

FROM python:3.8-slim-buster as base_dataset_image 

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install genometools -y --no-install-recommends

# -- Stage 2 -- #

FROM base_dataset_image as entrypoint_image

COPY . .
CMD python entrypoint.py