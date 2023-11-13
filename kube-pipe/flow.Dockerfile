# Produce un'immagine Docker per la compilazione delle pipeline Kubeflow.

FROM python:3.8-slim-buster

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]