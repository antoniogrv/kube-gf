## Dipendenze

- [Miniconda](https://conda.io/miniconda.html)
- [Docker](https://www.docker.com/)
- [Kubernetes](https://kubernetes.io/it/docs/concepts/overview/what-is-kubernetes/)
- [Kind](https://kind.sigs.k8s.io/)
- [Kubeflow](https://www.kubeflow.org/)
- [Kubeflow Pipelines (KFP Platform v2.0.2, KFP SDK v1.8.22)](https://www.kubeflow.org/docs/components/pipelines/v1/sdk/install-sdk/)

## Local Bootstrap

### Kubeflow

`kind create cluster --config kind-config/{x}n-c.yaml`

`kubectl cluster-info --context kind-kf3-cluster`

`kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.2"`

`kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io`

`kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=2.0.2"`

### Kubeflow Pipeline Port Forwarding

`kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`

### Registry Port Forwarding

Intra-cluster, è possibile accedere al registro mediante reg.kubeflow:5000. Fuori dal cluster, è possibile accedere al servizio tramite localhost:5000 una volta eseguito il port-forwarding.

`kubectl port-forward -n kubeflow svc/reg 5000:5000`
