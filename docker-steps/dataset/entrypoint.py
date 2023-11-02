from typing import Optional
from typing import Dict

from dotenv import load_dotenv

import os

from tokenizer import MyDNATokenizer
from tokenizer import DNABertTokenizer

from dataset import TranscriptDatasetConfig
from dataset import TranscriptDataset

from utils import define_gene_classifier_inputs


def entrypoint(
        type: str,
        csv_path: str,
        len_read: int,
        len_kmer: int,
        n_words: int,
        tokenizer_selected: str,
        model_selected: str,
        hyper_parameters: Dict[str, any],
        batch_size: int,
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

    dataset_conf: TranscriptDatasetConfig = TranscriptDatasetConfig(
        genes_panel_path='data/genes_panel.txt',
        transcript_dir='data/transcripts',
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer=tokenizer
    )

    if type == 'test':
        print('Preparing Test Dataset...')

        test_dataset = TranscriptDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='test'
        )

        test_dataset.get_dataset().to_csv(csv_path, index=False)

    if type == 'train':
        print('Preparing train Dataset...')

        train_dataset = TranscriptDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='train'
        )

        train_dataset.get_dataset().to_csv(csv_path, index=False)

    if type == 'val':
        print('Preparing Validation Dataset...')

        val_dataset = TranscriptDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='val'
        )

        val_dataset.get_dataset().to_csv(csv_path, index=False)

    return type


if __name__ == '__main__':
    __args, __hyper_parameters = define_gene_classifier_inputs()

    entrypoint(
        **__args,
        hyper_parameters=__hyper_parameters
    )
