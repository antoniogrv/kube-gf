# GeneFusion over Kubernetes

Questo progetto mira a realizzare un'architettura distribuita altamente scalabile per carichi di lavoro di machine learning. In particolare, il sistema è stato tarato per l'analisi di sequenze di DNA e RNA, e per l'individuazione di geni di fusione.

Il sistema prevede un'infrastruttura Kubernetes su cui è possibile eseguire pipeline di machine learning e deep learning, realizzata tramite Kubeflow e sviluppata localmente con Kind. L'obiettivo è rendere l'architettura quanto più portabile possibile, in modo da poter essere eseguita su qualsiasi infrastruttura Kubernetes, sia essa on-prem o cloud-based. Come possibile istanza di un problema che l'architettura potrebbe accogliere, il [modello monolitico GeneFusion](https://github.com/FLaTNNBio/gene-fusion-kmer) è stato scorporato in microservizi, ognuno dei quali è stato containerizzato e reso disponibile tramite un Docker Registry on-prem. Questi container, opportunamente orchestrati, convergono in una pipeline programmatica, robusta e sicura.

## Installazione del sistema

Il bootstrap del progetto prevede la creazione di un cluster Kubernetes e l'applicazione dei manifesti Kubeflow, così come previsto dalla documentazione ufficiale del progetto. Per le attività di sviluppo, si consiglia di usare Kind, un tool che permette di creare cluster Kubernetes containerizzati tramite Docker. 

Prima di procedere, assicurarsi di aver installato le seguenti dipendenze:

- [Docker](https://www.docker.com/)
- [Kind](https://kind.sigs.k8s.io/)
- [Miniconda](https://conda.io/miniconda.html)

L'uso di Miniconda non è obbligatorio ma altamente consigliato, poiché il progetto prevede svariati ambienti autonomi ognuno col proprio subset di dipendenze Python. Una volta installato Docker (si consiglia l'intera suite di Docker Desktop piuttosto che esclusivamente il Docker Daemon), sarà possibile installare la dipendenza di [Kubernetes]((https://kubernetes.io/it/docs/concepts/overview/what-is-kubernetes/)) come estensione del software. Quest'ultima dovrebbe fornire ed inserire nel path anche la CLI `kubectl`; se così non fosse, è possibile installarla manualmente.

Infine, sarà possibile procedere all'installazione dell'ambiente [Kubeflow](https://www.kubeflow.org/) [(KFP Platform v2.0.2, KFP SDK v1.8.22)]((https://www.kubeflow.org/docs/components/pipelines/v1/sdk/install-sdk/)) generando dapprima un cluster Kind, per poi applicare il manifesto Kustomize di Kubeflow. In particolare, è previsto che il cluster Kind venga generato mediante un script che inizializzi anche un [Docker Registry contestuale al cluster Kubernetes](https://kind.sigs.k8s.io/docs/user/local-registry/) e con cui è possibile interagire localmente.

E' possibile quindi eseguire `./kubeflow-pipeline/kind/kind.sh` per generare il cluster locale. Una volta creato, switchare il context di Kubernetes per referenziare l'ambiente Kind tramite `kubectl cluster-info --context kind-kind`.

Una volta tarata la CLI di Kubernetes sul cluster Kind, eseguire sequenzialmente i seguenti comandi per applicare i manifesti Kubeflow:

```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.2"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=2.0.2"
```

Una volta completati i passaggi precedenti, è possibile effettuare il port-forwarding delle Kubeflow Pipelines tramite `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80` e accedervi via `http://localhost:8080`.

Per concludere l'installazione, si invita a creare un ambiente Miniconda tramite `conda create -n kf python=3.8.18`, attivarlo con `conda activate kf` e installare le dipendenze Python necessarie tramite `pip install -r kubeflow-pipeline/requirements.txt`. Una volta fatto, sarà possibile interagire col sistema tramite `python kubeflow-pipeline/kmer-pipeline.py`.

## Eseguire gli esperimenti

Dopo aver accuratamente eseguiti gli step di installazione del sistema, è sufficiente eseguire in sequenza `npm run forward` e `npm run exec` per avviare una pipeline ed eseguire l'esperimento descritto in `kubeflow-pipelines/kmer-pipeline.py`. Il primo comando esegue il port-forwarding delle Kubeflow Pipelines, mentre il secondo esegue l'esperimento vero e proprio.

## Sviluppo della pipeline

*Il contenuto di questa sezione è temporaneo.*

### Interagire col Docker Registry on-prem

Il sistema, così come installato, genera un container Docker con un Registry locale che può essere usato per caricare le immagini dei componenti della pipeline. Per caricare un'immagine nel registry, è necessario prima taggarla con il nome del registry stesso, che è `localhost:5001`. Per esempio, se si volesse caricare l'immagine `kmer-component:latest`, è necessario eseguire `docker tag kmer-component:latest localhost:5001/kmer-component:latest`. Una volta fatto, è possibile caricare l'immagine nel registry tramite `docker push localhost:5001/kmer-component:latest`.

### Creazione dei componenti

A linee generali, un componente per la pipeline dev'essere prima di tutto un'immagine Docker. Per creare un'immagine Docker, è necessario creare un file `Dockerfile` che definisca il contenuto dell'immagine stessa. Fare riferimento alla directory `docker-steps` per alcuni esempi.

L'immagine Docker deve quindi essere opportunamente taggata e pushata sul Docker Registry di Kind; inoltre, sarà necessario creare un file `component.yaml` che definisca il componente stesso. Di seguito un esempio di manifesto per un componente.

```yaml
name: Step di esempio
description: Descrizione dello step di esempio

implementation:
  container:
    image: localhost:5001/hello-world:linux
```

Infine, sarà possibile iniettare il componente esemplificativo nella pipeline come segue:

```python
step_op = kfp.components.load_component_from_file("components/step/component.yaml")

@dsl.pipeline(name="kmer-pipeline") 
def kmer_pipeline():
    step = step_op()
```
