### TBA

- [Miniconda](https://conda.io/miniconda.html)
- [Docker](https://www.docker.com/)
- [Kubernetes](https://kubernetes.io/it/docs/concepts/overview/what-is-kubernetes/)
- [Kind](https://kind.sigs.k8s.io/)
- [Kubeflow](https://www.kubeflow.org/)
- [Kubeflow Pipelines (KFP Platform v2.0.2, KFP SDK v1.8.22)](https://www.kubeflow.org/docs/components/pipelines/v1/sdk/install-sdk/)

### Local Bootstrap

`kind create cluster --config kind-config/{x}n-c.yaml`

`kubectl cluster-info --context kind-kf3-cluster`

`kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.2"`

`kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io`

`kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=2.0.2"`

`kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`
