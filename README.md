<div align="center">
<img align="center" src="https://kubernetes.io/images/nav_logo2.svg" alt="logo" width="300">
    <h3>GeneFusion over Kubernetes</h3>
    <p><b>Index Keys:</b> <a href="https://ml-ops.org/">MLOps</a>, <a href="https://mlsecops.com/">MLSecOps</a>, <a href="https://kubernetes.io/it/">Kubernetes</a>, <a href="https://www.kubeflow.org/">Kubeflow</a>
</div>

- [Introduzione](#introduzione)
- [Installazione del sistema](#installazione-del-sistema)
  - [Dipendenze](#dipendenze)
  - [Provisioning](#provisioning)
- [Eseguire la pipeline](#eseguire-la-pipeline)
    - [Caricare le immagini Docker](#caricare-le-immagini-docker)
        - [Dataset Generation Docker Image](#dataset-generation-docker-image)
        - [Model Training & Testing Docker Image](#model-training--testing-docker-image)
    - [Compilare le pipeline](#compilare-le-pipeline)
      - [Compilazione con Docker](#compilazione-con-docker)
      - [Compilazione con Miniconda](#compilazione-con-miniconda)
    - [Caricare la pipeline su Kubeflow](#caricare-la-pipeline-su-kubeflow)
- [Sviluppo della pipeline](#sviluppo-della-pipeline)
  - [Interagire col Docker Registry](#interagire-col-docker-registry)
  - [Creazione dei componenti](#creazione-dei-componenti)
- [Considerazioni di MLSecOps](#considerazioni-di-mlsecops)
  - [Iniezione arbitraria di CDK in un pod](#iniezione-arbitraria-di-cdk-in-un-pod)
  - [Generare un cluster Kind contagiato da CDK](#generare-un-cluster-kind-contagiato-da-cdk)
  - [Kubernetes Enumeration con Kali e Metasploit](#kubernetes-enumeration-con-kali-e-metasploit)
  - [Control Plane Load Testing con Artillery](#control-plane-load-testing-con-artillery)
- [Benchmark delle prestazioni](#benchmark-delle-prestazioni)

<hr>

## Introduzione

Questo progetto di tesi magistrale mira a realizzare un'architettura distribuita altamente scalabile per carichi di lavoro di machine learning. In particolare, il sistema è stato tarato per l'analisi di sequenze di DNA e RNA, e per l'individuazione di geni di fusione.

Il sistema prevede un'infrastruttura [Kubernetes](https://kubernetes.io/) su cui è possibile eseguire pipeline di machine learning e deep learning, realizzata tramite [Kubeflow](https://www.kubeflow.org/) e sviluppata localmente con [Kind](https://kind.sigs.k8s.io/). L'architettura può essere eseguita su qualsiasi cluster Kubernetes, sia esso on-prem o cloud-native (e.g. [AWS EKS](https://aws.amazon.com/it/eks/)). Come possibile istanza di un problema che l'architettura potrebbe accogliere, i [modelli monolitici Gene Classifier e Fusion Classifier](https://github.com/FLaTNNBio/gene-fusion-kmer) sono stati scorporati in microservizi, ognuno dei quali è stato containerizzato e reso disponibile tramite un [Docker Registry self-hosted](https://hub.docker.com/_/registry). Questi container, opportunamente orchestrati, convergono in una pipeline programmatica, robusta e sicura.

## Installazione del sistema

Il bootstrap del progetto prevede la creazione di un cluster Kubernetes, l'applicazione dei manifesti Kubeflow e attività di validazione della correttezza dell'installazione. Per maggiori informazioni sull'architettura del sistema, referenziare la sezione apposita.

### Dipendenze

> **Questo progetto non è compatibile con ambienti Windows.** E' strettamente necessario utilizzare un ambiente Linux. Qualsiasi tentativo di far combaciare le dipendenze richieste su [WSL](https://learn.microsoft.com/it-it/windows/wsl/) potrebbe non produrre i risultati auspicati. Inoltre, si sconsiglia di utilizzare macchine virtuali che non supportino la virtualizzazione hardware della GPU (e.g. [VirtualBox](https://www.virtualbox.org/)) poiché potrebbero generare conflitti con l'installazione del sistema. Si consiglia, pertanto, di utilizzare un'installazione nativa di Linux, per quanto sia teoricamente possibile [abilitare l'accelerazione della GPU in distribuzioni WSL](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview).

Prima di procedere, assicurarsi di aver installato correttamente le seguenti tecnologie:
- [Docker](https://docs.docker.com/engine/install/) (v24.0.7)
- [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) (v0.20.0)
- [Helm](https://helm.sh/docs/intro/install/) (v3.13.1)
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) (Client v1.28.3, Server v1.27.3, Kustomize v5.0.4)
- [git](https://git-scm.com/)

Le versioni indicate nelle parentesi rappresentano quelle adoperate per lo sviluppo del sistema. A meno di forti deprecazioni, qualsiasi versione successiva dovrebbe garantire un'installazione corretta.

Inoltre, per garantire il supporto della GPU sono necessarie le seguenti dipendenze:
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
- [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/unix/)

> Su Ubuntu, non è necessario installare i driver NVIDIA poiché pre-installati autonomamente dalla distribuzione. Tuttavia, è necessario installare il container toolkit.

### Provisioning

1. Clonare la repository [kube-gf](https://github.com/antoniogrv/kube-gf) e accertarsi che il Docker Daemon sia in esecuzione.

```
git clone https://github.com/antoniogrv/kube-gf.git
cd kube-gf
sudo systemctl status docker
```

2. Eseguire lo script di provisioning del cluster locale Kubernetes con supporto per la GPU tramite la CLI di Kind; una volta creato, tarare la CLI di `kubectl` sul nuovo cluster Kind.

> L'esecuzione dei comandi da terminali di `docker`, `kind` e `kubectl` potrebbe richiedere i privilegi da amministratore (`sudo`).

```
chmod +x kube-pipe/kind/boot-kind-gpu.sh
./kube-pipe/kind/boot-kind-gpu.sh
kubectl cluster-info --context kind-kind
```

3. Verificare che il contenuto del file `/etc/docker/daemon.json` sia **esattamente** come segue. In caso di difformità, sostituire in toto il contenuto del file con quello indicato e riavviare il Docker Daemon con `sudo systemctl restart docker`.
```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  }
}
```

4. Riavviare il Docker Daemon. Successivamente, dare visione della GPU al nodo Kubernetes iniettando un file di configurazione all'interno del container. Si tratta di un workaround per [prevenire alcuni problemi ben noti](https://github.com/NVIDIA/nvidia-docker/issues/614#issuecomment-423991632).
```
docker exec -ti kind-control-plane ln -s /sbin/ldconfig /sbin/ldconfig.real
```

5. Installare il [Kubernetes Operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) per il rilevamento e l'etichettamento della disponibilità della GPU sui nodi ([NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator#nvidia-gpu-operator)). Quest'operazione potrebbe richiedere diverso tempo.
```
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
helm repo update
helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator --set driver.enabled=false
```

6. Applicare i manifesti [Kubeflow](https://www.kubeflow.org/) al cluster Kubernetes.
```
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.2"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-emissary?ref=2.0.2"
```

7. Poiché l'implementazione del NVIDIA Operator è talvolta imprevedibile, bisogna controverificare che stia operando come previsto. Assicurarsi che il NVIDIA GPU Operator sia effettivamente in esecuzione. Per farlo, consultare lo stato dei [pod](https://kubernetes.io/docs/concepts/workloads/pods/) nel [Kubernetes Namespace](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/) `gpu-operator`.
```
kubectl get pods -n gpu-operator
```
Qualora i pod dell'operatore non fossero in esecuzione (i.e. `CrashLoopBackOff`, `Error`, etc.), prendere nota dei nomi dei pod faulty (e.g. `nvidia-gpu-operator-xxx`) e distruggerli.
```
kubectl delete pod <pod-name> -n gpu-operator
```
I pod distrutti verranno ricreati automaticamente dall'operatore. Una volta confermato il funzionamento dei pod, monitorare i log del pod `nvidia-device-plugin-daemonset-xxx` (o simili) per verificare che la GPU sia stata correttamente rilevata sul nodo esemplificativo `kind-control-plane`. Il corretto funzionamento del DaemonSet è facilmente osservabile, poiché segnalerà di aver aggiornato le risorse del nodo con le GPU rilevate: a questo punto, controverificare che il numero delle GPU rilevate sia quanto atteso nel descrittore del nodo-container `kind-control-plane`; in particolare, confermare che in *Allocatable* sia indicato `nvidia.com/gpu: 1`.
```
kubectl logs -f nvidia-device-plugin-daemonset-xxx -n gpu-operator
kubectl describe node kind-control-plane
```

8. Una volta completati i passaggi precedenti, è possibile effettuare il port-forwarding delle Kubeflow Pipelines e accedervi da web browser via `http://localhost:8080`. 
```
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

## Eseguire la pipeline

> Di seguito vengono proposti gli step per eseguire la pipeline del modello *Gene Classifier*; similmente, è possibile eseguire la pipeline del modello *Fusion Classifier* in modo del tutto analogo.

Una volta creata l'infrastruttura Kubernetes, è possibile eseguire la pipeline esemplificativa presente in questa repository. Per farlo, è necessario caricare sul [Docker Registry](https://docs.docker.com/registry/) on-prem del cluster le immagini Docker (*docker-steps*) dei componenti della pipeline. Successivamente, si presentano due opzioni:

- Compilare manualmente la pipeline
- Caricare la pipeline come artefatto su Kubeflow

### Caricare le immagini Docker

I microservizi della pipeline giacciono in un registro Docker self-hosted all'interno del cluster Kubernetes. In particolare, sono stati realizzati due microservizi: uno per la generazione dei dataset, e uno per il training e il testing del modello con [PyTorch](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjguv6HxraCAxWGg_0HHbl7DagQFnoECAUQAQ&url=https%3A%2F%2Fpytorch.org%2F&usg=AOvVaw2mABY6VbqZdRJYnleMzDSb&opi=89978449) e [CUDA](https://developer.nvidia.com/cuda-toolkit). 

> Si noti che *non* è necessario installare Python, PyTorch, CUDA o altre dipendenze non espressamente indicate su questo README. Le immagini Docker sono state realizzate in modo tale da includere tutte le dipendenze necessarie per l'esecuzione dei componenti.

[I pod della pipeline Kubeflow effettuano il pull dal registro Docker](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/); analogamente, i componenti Kubeflow referenziano *esattamente* i tag delle immagini così come sono state caricate nel registro, motivo per cui è necessario creare le immagini, taggarle opportunamente e caricarle nel registro.

Sia il modello Gene Classifier che il modello Fusion Classifier condividono le due immagini Docker proposte, forti di un'opportuna parametrizzazione.

#### Dataset Generation Docker Image

Eseguire il seguente comando. 

```console
docker build -t localhost:5001/step-dataset-generation-config:latest ./docker-steps/dataset && \
docker push localhost:5001/step-dataset-generation-config:latest
```

#### Model Training & Testing Docker Image

Eseguire il seguente comando.

```console
docker build -t localhost:5001/step-model-config:latest ./docker-steps/model && \
docker push localhost:5001/step-model-config:latest
```

### Compilare le pipeline

Seguire le seguenti istruzioni solo se si intende compilare manualmente la pipeline. Se si intende usare la pipeline pre-compilata, saltare direttamente alla sezione successiva.

Sono stati progettate due modalità per compilare le pipeline.
- Usare Docker *(consigliato)*
- Usare [Miniconda](https://conda.io/miniconda.html)

> I seguenti esempi generano la pipeline per il modello Gene Classifier impiegando lo script `compile_gene_classifier_pipeline.py`. E' possibile generare la pipeline per il modello Fusion Classifier usando lo script `compile_fusion_classifier_pipeline.py`.

#### Compilazione con Docker

Eseguire il seguente comando. Il risultato sarà disponibile nella directory `kube-pipe/relics`.

```console
docker build -t flow -f kube-pipe/flow.Dockerfile kube-pipe && \
docker run -v ./kube-pipe/relics:/relics flow compile_gene_classifier_pipeline.py
```

#### Compilazione con Miniconda

1. [Installare Miniconda seguendo le istruzioni indicate sul sito](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

2. Creare e attivare l'ambiente Miniconda.

```
conda create -n kf python=3.8.18
conda activate kf
```

3. Installare le dipendenze Python necessarie.

```
pip install -r kube-pipe/requirements.txt
```

4. Eseguire uno dei due script di compilazione, il quale produrrà un artefatto nella directory `kube-pipe/relics`.
```
python kube-pipe/compile_gene_classifier_pipeline.py
```

### Caricare la pipeline su Kubeflow

- Se si decide di usare la pipeline compilata manualmente, il manifesto sarà disponibile localmente a `kube-pipe/relics/gene_classifier_pipeline.yaml`.
- Se si decide di usare la pipeline pre-compilata, sarà disponibile localmente a `artifacts/pipelines/gene_classifier_pipeline.yaml`.

A prescindere da come si preleva il file `gene_classifier_pipeline.yaml`, quest'ultimo dev'essere caricato su Kubeflow per poter generare esperimenti ed esecuzioni.

1. Dirigersi sulla dashboard Kubeflow all'indirizzo `http://localhost:8080`.
2. In *Pipelines*, cliccare su *Upload pipeline* e selezionare il file locale `gene_classifier_pipeline.yaml`.

> Nota: in alternativa, invece di caricare il file locale `gene_classifier_pipeline.yaml`, è possibile caricare il manifesto come URL remoto tramite `https://github.com/antoniogrv/kube-gf/releases/download/draft/gene_classifier_pipeline.yaml`. E' possibile trovare altri manifesti pre-compilati nella sezione [Releases](https://github.com/antoniogrv/kube-gf/releases).

3. Dovrebbe comparire un prospetto della macchina a stati. A questo punto, cliccare *Create run* per eseguire la pipeline.

## Sviluppo della pipeline

Questo progetto mira anche ad essere un punto di riferimento per i data scientists e developer che vogliono sfruttare le potenzialità di Kubernetes. Come ampiamente descritto dalla documentazione di Kubeflow, il tool permette di integrare componenti personalizzati (come, ad esempio, le due immagini Docker realizzate nell'ottica di questa attività di tesi magistrale), che fungeranno da *step* di una più vasta macchina a stati che rappresentano la spina dorsale delle pipeline di MLOps. 

### Interagire col Docker Registry

Il sistema, così come installato, genera due container Docker, di cui un Registry locale che può essere usato per caricare le immagini dei componenti della pipeline. Per caricare un'immagine nel registry, è necessario prima taggarla con il namespace del registry stesso, che è `localhost:5001`. Per esempio, se si volesse caricare l'immagine `kmer-component:latest`, sarebbe necessario eseguire `docker tag kmer-component:latest localhost:5001/kmer-component:latest`. Una volta fatto, sarebbe possibile caricare l'immagine nel registry tramite `docker push localhost:5001/kmer-component:latest`.

> Chiaramente, utilizzare un Docker Registry privato non è l'unica opzione, né è quella consigliata, eccetto che per motivi didattici e formativi. In un ambiente di produzione, si consiglia l'impiego di [AWS Elastic Container Registry (ECR)](https://aws.amazon.com/it/ecr/) o [GCP Artifact Registry](https://cloud.google.com/artifact-registry?hl=it).

A questo punto, l'immagine risulterebbe disponibile nel registry e sarebbe possibile iniettarla nella pipeline. Il container runtime dei nodi su cui è eseguito Kubeflow saranno in grado di effettuare il pull dell'immagine, recepirne l'interfaccia I/O e orchestrarne il ciclo di vita.

### Creazione dei componenti

A linee generali, un componente per la pipeline dev'essere prima di tutto un'immagine Docker. Per creare un'immagine Docker, è necessario creare un file `Dockerfile` che definisca il contenuto dell'immagine stessa. Fare riferimento alla directory `docker-steps` per ulteriori esempi, e alla [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/) per la documentazione ufficiale.

<details>
<summary>Model Training & Testing Docker Image Reference</summary>

```dockerfile
# Produce un'immagine Docker parametrica che effettua il training e il testing del modello.

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

COPY --from=localhost:5001/step-dataset-generation-config:latest transformers/ /transformers
COPY . .

FROM cuda as core

COPY --from=fs . .
COPY deps/transcript_label.pkl data/inputs_model/transcript_label.pkl
COPY deps/chimeric_label_fusion.pkl data/inputs_model/chimeric_label_fusion.pkl

ENTRYPOINT python3.9 gc_handler.py
```

</details>

L'immagine Docker deve quindi essere opportunamente taggata e pushata sul Docker Registry di Kind, in modo non dissimile da come illustrato nel paragrafo [Caricare le immagini Docker](#caricare-le-immagini-docker).

Infine, sarà necessario creare un file `component.yaml` che definisca il componente stesso. A linee generali, un componente ha una forma di questo tipo:

```yaml
name: Step di esempio
description: Descrizione dello step di esempio

implementation:
  container:
    image: localhost:5001/hello-world:linux
```

Di seguito manifesto esemplificativo per un componente.

```yaml
name: Fusion Classifier Model Training
description: Effettua l'addestramento del modello FC

inputs:
- {name: train_csv_path, type: String, description: 'Training CSV path.'}
- {name: val_csv_path, type: String, description: 'Validation CSV path.'}
- {name: gc_model_path, type: String, description: 'Absolute path of the Gene Classifier H5 model.'}

outputs:
- {name: model_path, type: String, description: 'H5 Model path.'}

implementation:
  container:
    image: localhost:5001/step-model-config:latest
    command: [
      python3.9,
      'fc_handler.py',
      -gc_model_path, 
      {inputPath: gc_model_path},
      -train_csv_path, 
      {inputPath: train_csv_path},
      -val_csv_path, 
      {inputPath: val_csv_path},
      -model_path, 
      {outputPath: model_path}
    ]
```

A questo punto, sarà possibile iniettare il componente esemplificativo nella pipeline come segue:

```python
import os
import kfp

from kfp import compiler
from kfp.v2 import dsl

'''
Delinea i componenti della pipeline Kubeflow.
'''
@dsl.pipeline(name="Gene Classifier") 
def gene_classifier():
    gc_dataset_train_config = gc_dataset_train_config_op()

    [..]

    gc_model_test_config = gc_model_test_config_op(
        test_csv_path=gc_dataset_test_config.outputs["csv_path"],
        model_path=gc_model_train_config.outputs["model_path"]
    ).set_gpu_limit(1)

if __name__ == '__main__':
    # Componenti Docker della pipeline Kubeflow
    gc_dataset_train_config_op = kfp.components.load_component_from_file(component_path("gene-classifier/dataset/train"))

    [..]

    gc_model_test_config_op = kfp.components.load_component_from_file(component_path("gene-classifier/model/test"))

    # Compilazione della pipeline
    compiler.Compiler().compile(gene_classifier, package_path=os.path.join(os.path.dirname(__file__), 'relics/gene_classifier_pipeline.yaml'))
```

Eseguire lo script Python produrrà il manifesto della pipeline, che potrà essere caricato su Kubeflow come descritto nella sezione [Caricare la pipeline su Kubeflow](#caricare-la-pipeline-su-kubeflow).

## Considerazioni di MLSecOps

Un'analisi broad di [Container Security](https://www.redhat.com/it/topics/security/container-security) è stata condotta sull'architettura presentata. Più specificamente, si intende approfondire le possibilità interazioni fra un avversario dotato di strumenti di attacco più o meno sofisticati, e il cluster Kubernetes dotato delle sue componenti Kubeflow così come presentato. Questa operazione si inserisce in un più vasto contesto di studi rappresentato dalla branca [MLSecOps](https://mlsecops.com/).

Gli ingredienti di questa analisi si declinano nelle seguenti tecnologie:
- [Kali Linux Container](https://www.kali.org/)
- [Zero-dependency Container Penetration Toolkit (CDK)](https://github.com/cdk-team/CDK)
- [Metasploit Framework](https://docs.metasploit.com/)

In particolare, si intendono indagare le seguenti applicazioni:

- Dall'*esterno*, immaginare un host avversario che tenta di compromettere il cluster Kubernetes. In questo caso, l'host avversario è rappresentato da un container Kali Linux, che monta il toolkit CDK e Metasploit Framework. L'obiettivo è quello di tracciare le difese del cluster, e in particolare del nodo Kubernetes `kind-control-plane`.
- Dall'*interno*, immaginare un avvesario che ha già compromesso il cluster Kubernetes. In questo caso, l'avversario è in grado di tracciare le vulnerabilità e il perimetro di difesa del nodo Kubernetes (su cui giacciono i pod Kubeflow) mediante il toolkit CDK, ed in particolare gli exploit di Information Gathering. In questo caso, l'obiettivo è individuare informazioni sensibili da poter sfruttare per un escalation.

> Per eseguire l'analisi di sicurezza presentata, non è necessario installare ulteriori dipendenze rispetto a quelle già definite in precedenza. E' sufficiente che il Docker Daemon sia in esecuzione.

### Iniezione arbitraria di CDK in un pod

E' stata realizzata un'immagine Docker che monta il toolkit CDK su base [Alpine](https://alpinelinux.org/). Per costruirla e caricarla sul Docker Registry on-prem, eseguire il seguente comando.

```console
docker build -t localhost:5000/cdk --file=container-sec/cdk.Dockerfile container-sec && \
docker push localhost:5001/step-dataset-generation-config:latest
```

L'mmagine può essere iniettata arbitrariamente nel cluster Kubernetes, sul nodo `kind-control-plane`, producendo un nuovo pod all'interno dello stesso, col fine ultimo di eseguire attacchi di Penetration Testing. Per farlo, eseguire il seguente comando.

```
kubectl debug node/kind-control-plane -it --image=localhost:5001/cdk
```

Questo approccio, per quanto indagato, non è stato approfondito ulteriormente, poiché strettamente vincolato alla capacità dell'avversario di iniettare immagini corrotta nel registro, ed eseguire un pull malizioso.

### Generare un cluster Kind contagiato da CDK

Durante l'installazione del sistema, al paragrafo [Provisioning](#provisioning), è stato eseguito uno shell script (`boot-kind-gpu.sh`) per generare il cluster Kubernetes con Kind. Tale script, per rappresentare i nodi del cluster, utilizza l'immagine Docker [kindest/node](https://hub.docker.com/r/kindest/node/). E' invece possibile generare un cluster Kind corrotto, popolato da nodi infetti col toolkit CDK, sfruttando un'immagine Docker personalizzata chiamata `kind-cdk` e realizzata contestualmente a questo lavoro di tesi. Costruendo il cluster in questo modo, è possibile indagare le difese del cluster Kubernetes, e in particolare del nodo `kind-control-plane` infetto, eseguendo valutazioni di Information Gathering con CDK.

Per costruire l'immagine Docker `kind-cdk`, eseguire il seguente comando.

```console
docker build -t kind-cdk --file=container-sec/kind.Dockerfile container-sec
```

A questo punto, è possibile reiterare le stesse operazioni descritte nel capitolo [Installazione del sistema](#installazione-del-sistema), avendo però cura di utilizzare lo shell script `boot-kind-gpu-cdk.sh` invece di `boot-kind-gpu.sh`. Lo script produrrà un cluster Kind chiamato `kind-cdk` associabile a `kubectl` col comando `kubectl cluster-info --context kind-kind-cdk`.

Conclusa l'installazione, sarà possibile accedere al control plane di Kubernetes col seguente comando, per poi confermare che CDK sia attivo usando il comando `cdk`.

```console
docker exec -it kind-control-plane sh
```

E' adesso possibile individuare le debolezze note del container con `cdk eva`. E' immediantamente osservabile che il nodo risulta vulnerabile sotto molteplici aspetti, così come segnalato dal toolkit. Ad esempio, risultano esposte alcune variabili d'ambiente e file di configurazione che potrebbero essere sfruttate per un escalation.

```log
[  Information Gathering - Services  ]
2023/11/13 11:49:46 sensitive env found: container=docker
2023/11/13 11:49:46 sensitive env found: KUBECONFIG=/etc/kubernetes/admin.conf

[..]

cat /etc/kubernetes/admin.conf

users:
- name: kubernetes-admin
  user:
    client-certificate-data: [..]
    client-key-data: [..]
```

Il file di log prodotto da `cdk eva` a partire da un nodo Kind vergine è disponibile nella directory `container-sec/logs` di questa repository.

### Kubernetes Enumeration con Kali e Metasploit

Una versione personalizzata di Kali Linux (che monta, fra le altre cose, il toolkit CDK) può essere prodotta a partire dal Dockerfile `kali.Dockerfile` presente nella directory `container-sec`. Per farlo, eseguire il seguente comando.

```console
docker build -t kali-cdk --file=container-sec/kali.Dockerfile container-sec
```

Per eseguire questa versione di Kali Linux, è sufficiente eseguire:

```console
docker run --network=host -i --tty kali-cdk
```

Di seguito la costruzione del `kali.Dockerfile`, comprensivo di CDK e delle dipendenze necessarie per l'utilizzo di Metasploit Framework.

<details>
<summary>Kali Linux Docker Image</summary>

```dockerfile
# Produce un'immagine Docker di Kali Linux per le analisi di Container Security e l'esecuzione di attacchi di Penetration Testing.
# Fare riferimento alla documentazione del progetto per maggiori dettagli.

FROM kalilinux/kali-rolling@sha256:b6bd6a9e6f62171e4e0d8f43f4b0d02f1df0ba493225da852968576bc2d602d2

WORKDIR /root
ENV DEBIAN_FRONTEND=noninteractive

# L'immagine di Kali Linux non include di default il pacchetto kali-linux-headless ..
# .. che contiene la maggior parte degli strumenti di analisi e attacco.
# Rif. https://www.kali.org/docs/containers/using-kali-docker-images/

RUN apt -y update && \
    apt -y dist-upgrade && \
    apt -y autoremove && \
    apt clean && \
    apt -y install wget

# Siamo interessati esclusivamente a Metasploit Framework, contenuto nel pacchetto kali-linux-top10.
# Immagini più organiche potrebbero utilizzare il pacchetto kali-linux-all.
# Rif. https://www.kali.org/blog/kali-linux-metapackages/

RUN apt -y install -f kali-tools-top10

# Iniezione arbitraria del binario del Zero Dependency Container Penetration Toolkit
# Rif. https://github.com/cdk-team/CDK

RUN wget https://github.com/cdk-team/CDK/releases/download/v1.5.2/cdk_linux_amd64 && \
    chmod a+x cdk_linux_amd64 && \
    mv cdk_linux_amd64 /usr/local/bin/cdk
```

</details>

Il container Kali verrà utilizzato prevalentemente come di appoggio per il software [Metasploit](https://docs.metasploit.com/), tool di exploiting e penetration testing ben noto nel settore InfoSec. In particolare, verranno utilizzati i moduli di [Kubernetes Penetration Testing](https://docs.metasploit.com/docs/pentesting/metasploit-guide-kubernetes.html) al fine di esporre le vulnerabilità del cluster Kind. Si tratta, in particolare, dei seguenti moduli:
- [modules/auxiliary/cloud/kubernetes/enum_kubernetes](https://github.com/rapid7/metasploit-framework/blob/master/documentation/modules/auxiliary/cloud/kubernetes/enum_kubernetes.md)
- [modules/exploit/multi/kubernetes/exec](https://github.com/rapid7/metasploit-framework/blob/master/documentation/modules/exploit/multi/kubernetes/exec.md)

Per impiegare questi moduli, bisognerà prima di tutto generare un Service Account da affibbiare ad un ipotetico amministratore del cluster compromesso. Questa procedura è delinata nella sezione [Access Clusters Using the Kubernetes API](https://kubernetes.io/docs/tasks/administer-cluster/access-cluster-api/#without-kubectl-proxy) della documentazione di Kubernetes. In particolare, è sufficiente eseguire i seguenti comandi:
  
```
kubectl create -n default serviceaccount admin-sa --dry-run=client -o yaml | kubectl apply -f -
kubectl create -n default clusterrolebinding admin-sa-binding --clusterrole=cluster-admin --serviceaccount=default:admin-sa --dry-run=client -o yaml | kubectl apply -f -
kubectl create token admin-sa
```

L'ultimo comando dovrebbe restituire un [JSON Web Token](https://jwt.io/) (JWT) necessario per autenticarsi al cluster Kubernetes. Il JWT dovrebbe avere una forma di questo tipo:

<details>
<summary>K8s API Server Cluster Admin Dummy JWT</summary>

```bash
eyJhbGciOiJSUzI1NiIsImtpZCI6IjFHQW1LbjY3U2kyNTZod0s4Q2VldWtyYnRiM2Q0WnpiYU41dzU2d3MtdG8ifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImRlZmF1bHQtdG9rZW4iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiZGVmYXVsdCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6IjIzZDZmYTUxLTUyMjUtNDhiMS05ZTQwLWYxMjkwNDczY2ZmMyIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpkZWZhdWx0OmRlZmF1bHQifQ.bZm3byRsInnfHwv7XMCgBcVCfHSeIiyZGpD0OyFTerlbH60SYGlydFcFTMyNAyQMxp9hCGKqp901Ebv27eXnjGB6B7RFr3LhFQULw04ZCrcs29yv7UttSH2dBcX_GilFB9YZqlAws5cCsN31XHAC6XXVsGrCLvYhZqGiyvCeViOVSf-Pe0uSbRwycQ_Wok7bcrPn06SD89WtZRRN5PG14X9YxRv3Pojn-Tb4iA5U31HBx9vrHKJesvvPfkKUUC_7NJs7uia6up6zikbEVbXXJ76a6SsUT6zoVX13-ROlztC9R8dFi8S9sM8UjeGTL7rsP2YsAm6HTeSLrObBznzrrw
```

</details>


A questo punto, è possibile utilizzare il modulo `enum_kubernetes` per enumerare le risorse del cluster, e in particolare i pod.
1. Accedere a Kali con `docker run --network=host -i --tty kali-cdk`
2. Accedere alla shell di Metasploit digitando `msfconsole`
3. Accedere al modulo di exploit digitando `use auxiliary/scanner/http/kubernetes_enum`.
4. Impostare l'endpoint dell'API Server di Kubernetes con `set RHOST https://127.0.0.1:39175`
5. Impostare il JWT digitando `set TOKEN <JWT>` e sostituendo `<JWT>` con il JWT generato in precedenza
6. Eseguire l'exploit con `run`

Il software dovrebbe restituire *esattamente* le stesse risorse che si otterrebbero con `kubectl get all --all-namespaces`, mappando 1:1 l'infrastruttura Kubernetes presente sul sistema. Un log di esempio è disponibile in `container-sec/logs`.


### Control Plane Load Testing con Artillery

Un attacco di load testing con [Artillery](https://www.artillery.io/docs) può essere condotto sul control plane di Kubernetes. Per farlo, eseguire il seguente comando, che avvierà un container Docker e alla fine del testing restituirà le metriche di esecuzione in `container-sec/logs`.

```console
docker run --rm -it --network=host -v ${PWD}/container-sec:/container-sec artilleryio/artillery:latest run --insecure -o /container-sec/logs/artillery-results.csv /container-sec/k8s-api-blitz.yaml
```
Questo tipo di attacco può essere particolarmente utile per valutare la resilienza del control plane, e la sua esigenza di essere ridondato e posto davanti ad un load balancer.

## Benchmark delle prestazioni

La repository include un Dockerfile con cui è possibile costruire un'immagine Docker dei modelli GeneFusion. Per farlo, eseguire il seguente comando:

```console
docker build -t gc-monolith --file=kube-pipe/monolith.Dockerfile .
```

Sarà quindi possibile eseguire un'esecuzione completa (inclusiva di tutti gli step della macchina a stati della pipeline, fra cui training e testing del modello) del modello Gene Classifier con il seguente comando.

```console
docker run -it --gpus all gc-monolith
```

In generale, è possibile osservare che l'implementazione a microservizi riduce concretamente i tempi di esecuzione.
- **Architettura a microservizi:** 12m33s
- **Architettura monolitica:** 15m19s (+22.05%)

Si noti che questi dati sono basati su esecuzioni *dummy* tarate ad una singola epoca di training.