from typing import Optional
from typing import Dict

from dotenv import load_dotenv

import argparse
import os

from tokenizer import MyDNATokenizer
from tokenizer import DNABertTokenizer

from dataset import TranscriptDatasetConfig
from dataset import TranscriptDataset

from dataset import FusionDatasetConfig
from dataset import FusionDataset

from utils import define_gene_classifier_inputs
from utils import define_fusion_classifier_inputs


def fusion_classifier_entrypoint(
        type: str,
        csv_path: str,
        len_read: int,
        len_kmer: int,
        n_words: int,
        tokenizer_selected: str,
        n_fusion: int,
        gc_model_selected: str,
        gc_hyperparameters: Dict[str, any],
        gc_batch_size: int,
        gc_re_train: bool,
        model_selected: str,
        fc_hyper_parameters: Dict[str, any],
        batch_size: int,
        freeze: bool,
        re_train: bool,
        grid_search: bool,
):
    print("CSV Path", csv_path)

    save_dir = os.path.dirname(csv_path)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if type not in ['test', 'train', 'val']:
        raise ValueError('Tipologia di dataset non valido')

    # get value from .env
    root_dir = 'data'

    # init tokenizer
    tokenizer: Optional[MyDNATokenizer] = None

    if tokenizer_selected == 'dna_bert':
        tokenizer = DNABertTokenizer(
            root_dir=root_dir,
            len_kmer=len_kmer,
            add_n=False
        )
    elif tokenizer_selected == 'dna_bert_n':
        tokenizer = DNABertTokenizer(
            root_dir=root_dir,
            len_kmer=len_kmer,
            add_n=True
        )
    
    print('Preparing Dataset Configuration...')    

    dataset_conf: FusionDatasetConfig = FusionDatasetConfig(
        genes_panel_path='data/genes_panel.txt',
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer=tokenizer,
        n_fusion=n_fusion,
    )

    if type == 'test':
        print('Preparing Test Dataset...')

        test_dataset = FusionDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='test'
        )

        test_dataset.get_dataset().to_csv(csv_path, index=False)

    if type == 'train':
        print('Preparing train Dataset...')

        train_dataset = FusionDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='train'
        )

        train_dataset.get_dataset().to_csv(csv_path, index=False)

    if type == 'val':
        print('Preparing Validation Dataset...')

        val_dataset = FusionDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='val'
        )

        val_dataset.get_dataset().to_csv(csv_path, index=False)

    return type


if __name__ == '__main__':
    __args, __gc_hyperparameters, __hyperparameters = define_fusion_classifier_inputs()

    fusion_classifier_entrypoint(
        **__args,
        gc_hyperparameters=__gc_hyperparameters,
        fc_hyper_parameters=__hyperparameters
    )
