import os
import kfp

from kfp import compiler
from kfp.v2 import dsl

'''
Delinea i componenti della pipeline Kubeflow.
'''
@dsl.pipeline(name="Fusion Classifier") 
def fusion_classifier():
    # Sub-pipeline Gene Classifier
    gc_dataset_train_config = gc_dataset_train_config_op()
    gc_dataset_val_config = gc_dataset_val_config_op()

    gc_model_train_config = gc_model_train_config_op(
        train_csv_path=gc_dataset_train_config.outputs["csv_path"],
        val_csv_path=gc_dataset_val_config.outputs["csv_path"]
    ).set_gpu_limit(1)

    # Sub-pipeline Fusion Classifier
    fc_dataset_train_config = fc_dataset_train_config_op()
    fc_dataset_test_config = fc_dataset_test_config_op()
    fc_dataset_val_config = fc_dataset_val_config_op()

    fc_model_train_config = fc_model_train_config_op(
        gc_model_path=gc_model_train_config.outputs["model_path"],
        train_csv_path=fc_dataset_train_config.outputs["csv_path"],
        val_csv_path=fc_dataset_val_config.outputs["csv_path"]
    ).set_gpu_limit(1)

    fc_model_test_config = fc_model_test_config_op(
        gc_model_path=gc_model_train_config.outputs["model_path"],
        test_csv_path=fc_dataset_test_config.outputs["csv_path"],
        model_path=fc_model_train_config.outputs["model_path"]
    ).set_gpu_limit(1)

'''
Genera un path qualificato per il componente specificato.
'''
def component_path(dir: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"components/{dir}/component.yaml")

'''
Compila la pipeline, producendo un file .yaml che può essere eseguito da Kubeflow.
'''
if __name__ == '__main__':
    # Componenti Docker della pipeline Kubeflow

    # Sub-pipeline Gene Classifier
    gc_dataset_train_config_op = kfp.components.load_component_from_file(component_path("gene-classifier/dataset/train"))
    gc_dataset_val_config_op = kfp.components.load_component_from_file(component_path("gene-classifier/dataset/val"))

    gc_model_train_config_op = kfp.components.load_component_from_file(component_path("gene-classifier/model/train"))

    # Sub-pipeline Fusion Classifier
    fc_dataset_train_config_op = kfp.components.load_component_from_file(component_path("fusion-classifier/dataset/train"))
    fc_dataset_test_config_op = kfp.components.load_component_from_file(component_path("fusion-classifier/dataset/test"))
    fc_dataset_val_config_op = kfp.components.load_component_from_file(component_path("fusion-classifier/dataset/val"))

    fc_model_train_config_op = kfp.components.load_component_from_file(component_path("fusion-classifier/model/train"))
    fc_model_test_config_op = kfp.components.load_component_from_file(component_path("fusion-classifier/model/test"))

    # Compilazione della pipeline
    compiler.Compiler().compile(fusion_classifier, package_path=os.path.join(os.path.dirname(__file__), 'relics/fusion_classifier_pipeline.yaml'))