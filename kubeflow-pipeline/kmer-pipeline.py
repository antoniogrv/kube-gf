import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2 import compiler
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
)
from kubernetes import client as k8s_client

dataset_train_config_op = kfp.components.load_component_from_file("components/step-dataset-train-config/component.yaml")

@dsl.pipeline(
    name="kmer-pipeline",
) 
def kmer_pipeline():
    dataset_train_config = dataset_train_config_op()

client = kfp.Client()
#compiler.Compiler().compile(kmer_pipeline, package_path='pipeline.yaml')
client.create_run_from_pipeline_func(kmer_pipeline, arguments={})