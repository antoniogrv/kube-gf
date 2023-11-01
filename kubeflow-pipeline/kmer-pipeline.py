import os
import kfp

from kubernetes import client as k8s_client
from kfp import compiler
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import (Input, Output, Artifact, Dataset)

'''
Delinea i componenti della pipeline Kubeflow.
'''
@dsl.pipeline(name="kmer-pipeline") 
def kmer_pipeline():
    dataset_train_config = dataset_train_config_op()
    dataset_test_config = dataset_test_config_op()
    dataset_val_config = dataset_val_config_op()

'''
Genera un path assoluto per il componente specificato, in modo da poter eseguire (npm run exec) lo script dalla root del progetto.
'''
def component_path(dir: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"components/{dir}/component.yaml")

if __name__ == '__main__':
    # Componenti Docker della pipeline Kubeflow
    dataset_train_config_op = kfp.components.load_component_from_file(component_path("step-dataset-train-config"))
    dataset_test_config_op = kfp.components.load_component_from_file(component_path("step-dataset-test-config"))
    dataset_val_config_op = kfp.components.load_component_from_file(component_path("step-dataset-val-config"))

    # Compilazione della pipeline
    compiler.Compiler().compile(kmer_pipeline, package_path=os.path.join(os.path.dirname(__file__), 'artifacts/pipeline.yaml'))

    # Esecuzione della pipeline
    client = kfp.Client()
    client.create_run_from_pipeline_func(kmer_pipeline, arguments={})