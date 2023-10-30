from typing import Optional
from typing import Dict

from dotenv import load_dotenv
import numpy as np
import os

from tokenizer import MyDNATokenizer
from tokenizer import DNABertTokenizer

from dataset import TranscriptDatasetConfig
from dataset import TranscriptDataset
from torch.utils.data import DataLoader

from utils import SEPARATOR
from utils import define_gene_classifier_inputs


def entrypoint(
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

    print('Preparing Test Dataset...')

    test_dataset = TranscriptDataset(
        root_dir=root_dir,
        conf=dataset_conf,
        dataset_type='test'
    )

    return test_dataset


def call():
    __args, __hyper_parameters = define_gene_classifier_inputs()

    load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

    entrypoint(
        **__args,
        hyper_parameters=__hyper_parameters
    )
