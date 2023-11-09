import os
import kfp

from kfp import compiler
from kfp.v2 import dsl

'''
Delinea i componenti della pipeline Kubeflow.
'''
@dsl.pipeline(name="gene_classifier") 
def gene_classifier():
    dataset_train_config = dataset_train_config_op()
    dataset_test_config = dataset_test_config_op()
    dataset_val_config = dataset_val_config_op()

    model_train_config = model_train_config_op(
        train_csv_path=dataset_train_config.outputs["csv_path"],
        val_csv_path=dataset_val_config.outputs["csv_path"]
    ).set_gpu_limit(1)

    model_test_config = model_test_config_op(
        test_csv_path=dataset_test_config.outputs["csv_path"],
        model_path=model_train_config.outputs["model_path"]
    ).set_gpu_limit(1)

'''
Genera un path assoluto per il componente specificato, in modo da poter eseguire (npm run exec) lo script dalla root del progetto.
'''
def component_path(dir: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"components/gene-classifier/{dir}/component.yaml")

'''
Compila la pipeline, producendo un file .yaml che può essere eseguito da Kubeflow.
'''
if __name__ == '__main__':
    # Componenti Docker della pipeline Kubeflow
    dataset_train_config_op = kfp.components.load_component_from_file(component_path("step-dataset-train-config"))
    dataset_test_config_op = kfp.components.load_component_from_file(component_path("step-dataset-test-config"))
    dataset_val_config_op = kfp.components.load_component_from_file(component_path("step-dataset-val-config"))

    model_train_config_op = kfp.components.load_component_from_file(component_path("step-model-train-config"))
    model_test_config_op = kfp.components.load_component_from_file(component_path("step-model-test-config"))

    # Compilazione della pipeline
    compiler.Compiler().compile(gene_classifier, package_path=os.path.join(os.path.dirname(__file__), 'relics/gene_classifier_pipeline.yaml'))

    # Esecuzione della pipeline
    #client = kfp.Client()
    #client.create_run_from_pipeline_func(kmer_pipeline, arguments={})