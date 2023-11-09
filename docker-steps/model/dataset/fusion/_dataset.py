from typing import Hashable
from pandas import Series
from typing import Union
from typing import Tuple
from typing import List
from typing import Dict

from tabulate import tabulate
import pandas as pd
import numpy as np
import pickle
import dotenv
import torch
import os

from multiprocessing.pool import Pool
from functools import partial

from dataset.fusion import FusionDatasetConfig
from dataset import MyDataset
from torch.utils.data.dataset import T_co

from sklearn.model_selection import train_test_split

from dataset.utils import split_dataset_on_processes
from dataset.utils import generate_kmers_from_dataset
from dataset.utils import generate_sentences_encoded_from_dataset

from dataset.utils import fusion_simulator
from dataset.utils import art_illumina
from dataset.utils import generate_reads


class FusionDataset(MyDataset):
    def __init__(
            self,
            root_dir: str,
            conf: FusionDatasetConfig,
            dataset_type: str,
            train_csv_path: str,
            test_csv_path: str,
            val_csv_path: str,
    ):
        # call super class
        super().__init__(
            root_dir=root_dir,
            check_dir_name='check',
            check_dict_name='fusion_dataset',
            conf=conf,
            dataset_type=dataset_type
        )

        # ======================== Load Gene Panel ======================== #
        self.__gene_panel_path: str = conf.genes_panel_path
        with open(self.__gene_panel_path, 'r') as gene_panel_file:
            self.__genes_list: List[str] = gene_panel_file.read().split('\n')
        self.update_file(self.__gene_panel_path)

        __train_dataset_path: str = train_csv_path
        __val_dataset_path: str = val_csv_path
        __test_dataset_path: str = test_csv_path

        if dataset_type == 'test': self.__dataset_path = test_csv_path
        if dataset_type == 'train': self.__dataset_path = train_csv_path
        if dataset_type == 'val': self.__dataset_path = val_csv_path

        self.__dataset: pd.DataFrame = pd.read_csv(self.__dataset_path)
        self.__status = self.__dataset.groupby('label')['label'].count()

        # ==================== Create inputs for model ==================== #
        self.__inputs_path: str = os.path.join(
            self.inputs_dir,
            f'chimeric_{conf.len_read}_'
            f'{conf.n_fusion}_'
            f'kmer_{conf.len_kmer}_'
            f'n_words_{conf.n_words}_'
            f'tokenizer_{conf.tokenizer}_'
            f'fusion_'
            f'{self.dataset_type}.pkl'
        )
        # check if inputs tensor are already generateds
        generation_inputs_phase: bool = self.check_file(self.__inputs_path)
        print('Generation input phase is', generation_inputs_phase)
        if not generation_inputs_phase:
            # get number of processes
            n_proc: int = 1
            # get number of kmers and number of sentences
            __n_kmers: int = conf.len_read - conf.len_kmer + 1
            __n_sentences: int = __n_kmers - conf.n_words + 1
            # init inputs
            self.__inputs: [Dict[str, Union[List[Dict[str, torch.Tensor]], torch.Tensor]]] = []
            # check if n_proc is greater then 1
            if n_proc == 1:
                # call generate sentences encoded from dataset on single process
                self.__inputs = generate_sentences_encoded_from_dataset(
                    rows_index=(0, len(self.__dataset)),
                    dataset=self.__dataset,
                    n_words=conf.n_words,
                    n_kmers=__n_kmers,
                    n_sentences=__n_sentences,
                    tokenizer=conf.tokenizer
                )
            # TODO: debug the call of this function in parallel
            else:
                # split dataset on processes
                rows_for_each_process: List[Tuple[int, int]] = split_dataset_on_processes(
                    self.__dataset,
                    n_proc
                )
                # call generate sentences encoded from dataset on multi processes
                with Pool(n_proc) as pool:
                    results = pool.imap(partial(
                        generate_sentences_encoded_from_dataset,
                        dataset=self.__dataset,
                        n_words=conf.n_words,
                        n_kmers=__n_kmers,
                        n_sentences=__n_sentences,
                        tokenizer=conf.tokenizer
                    ), rows_for_each_process)
                    # append all local inputs to global dataset
                    for local_inputs in results:
                        self.__inputs += local_inputs
            with open(self.__inputs_path, 'wb') as handle:
                print('Picke-loading data...')
                pickle.dump(self.__inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.update_file(self.__inputs_path)
        # load inputs
        else:
            with open(self.__inputs_path, 'rb') as handle:
                self.__inputs: [Dict[str, Union[List[Dict[str, torch.Tensor]], torch.Tensor]]] = pickle.load(handle)

    def get_labels_dict(self) -> Dict[str, int]:
        return self.__labels

    def get_dataset_status(self):
        return self.__status

    def print_dataset_status(self) -> str:
        table: List[List[Union[Hashable, Series]]] = [[label, record] for label, record in self.__status.items()]
        table_str: str = tabulate(
            tabular_data=table,
            headers=['label', 'no. records'],
            tablefmt='psql'
        )
        return f'\n{table_str}\n'

    def classes(self):
        return len(self.__labels.keys())

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, index) -> T_co:
        return self.__inputs[index]
