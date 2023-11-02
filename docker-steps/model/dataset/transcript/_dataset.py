from typing import Hashable
from pandas import Series
from typing import Union
from typing import Tuple
from typing import Dict
from typing import List

from tabulate import tabulate
import pandas as pd
import pickle
import torch
import os

from multiprocessing.pool import Pool
from functools import partial

from dataset.transcript import TranscriptDatasetConfig
from dataset import MyDataset
from torch.utils.data.dataset import T_co

from sklearn.model_selection import train_test_split

from dataset.utils import split_dataset_on_processes
from dataset.utils import split_reads_file_on_processes
from dataset.utils import generate_sentences_from_kmers
from dataset.utils import generate_kmers_from_sequences
from dataset.utils import encode_sentences

from dataset.utils import gt_shredder


class TranscriptDataset(MyDataset):
    def __init__(
            self,
            root_dir: str,
            conf: TranscriptDatasetConfig,
            dataset_type: str
    ):
        # call super class
        super().__init__(
            root_dir=root_dir,
            check_dir_name='check',
            check_dict_name='transcript_dataset',
            conf=conf,
            dataset_type=dataset_type
        )

        print('Reading labels...')
        self.__labels_path: str = os.path.join(self.inputs_dir, 'transcript_label.pkl')
        with open(self.__labels_path, 'rb') as handle:
                self.__labels: Dict[str, int] = pickle.load(handle)

        __train_dataset_path: str = os.path.join(
            self.processed_dir,
            f'transcript_{conf.len_read}_'
            f'kmer_{conf.len_kmer}_'
            f'n_words_{conf.n_words}_'
            f'train.csv'
        )
        __val_dataset_path: str = os.path.join(
            self.processed_dir,
            f'transcript_{conf.len_read}_'
            f'kmer_{conf.len_kmer}_'
            f'n_words_{conf.n_words}_'
            f'val.csv'
        )
        __test_dataset_path: str = os.path.join(
            self.processed_dir,
            f'transcript_{conf.len_read}_'
            f'kmer_{conf.len_kmer}_'
            f'n_words_{conf.n_words}_'
            f'test.csv'
        )
        print('Checking datasets...')
        # check if train, val and test set are already generateds
        generation_sets_phase_flag: bool = (
                self.check_dataset(__train_dataset_path) and
                self.check_dataset(__val_dataset_path) and
                self.check_dataset(__test_dataset_path)
        )
        if not generation_sets_phase_flag:
            print('Datasets are corrupted.')
        else:
            print('Datasets look fine.')
        # load dataset
        self.__dataset_path = os.path.join(
            self.processed_dir,
            f'transcript_{conf.len_read}_'
            f'kmer_{conf.len_kmer}_'
            f'n_words_{conf.n_words}_'
            f'{self.dataset_type}.csv'
        )
        self.__dataset: pd.DataFrame = pd.read_csv(self.__dataset_path)
        self.__status = self.__dataset.groupby('label')['label'].count()

        # ==================== Create inputs for model ==================== #
        self.__inputs_path: str = os.path.join(
            self.inputs_dir,
            f'transcript_{conf.len_read}_'
            f'kmer_{conf.len_kmer}_'
            f'n_words_{conf.n_words}_'
            f'tokenizer_{conf.tokenizer}_'
            f'{self.dataset_type}.pkl'
        )
        # check if inputs tensor are already generateds
        generation_inputs_phase: bool = generation_sets_phase_flag and self.check_file(self.__inputs_path)

        print('Generation input phase is', generation_inputs_phase)

        if not generation_inputs_phase:
            # get number of processes
            n_proc: int = 1
            # init inputs
            self.__inputs: List[Dict[str, torch.Tensor]] = []
            # check if n_proc is greater then 1
            if n_proc == 1:
                # call encode sentences on single process
                self.__inputs: List[Dict[str, torch.Tensor]] = encode_sentences(
                    rows_index=(0, len(self.__dataset)),
                    dataset=self.__dataset,
                    n_words=conf.n_words,
                    tokenizer=conf.tokenizer
                )
            else:
                # split dataset on processes
                rows_for_each_process: List[Tuple[int, int]] = split_dataset_on_processes(
                    self.__dataset,
                    n_proc
                )
                # call encode sentences on multi processes
                with Pool(n_proc) as pool:
                    results = pool.imap(partial(
                        encode_sentences,
                        dataset=self.__dataset,
                        n_words=conf.n_words,
                        tokenizer=conf.tokenizer
                    ), rows_for_each_process)
                    # append all local inputs to global inputs
                    for local_inputs in results:
                        self.__inputs += local_inputs
            with open(self.__inputs_path, 'wb') as handle:
                pickle.dump(self.__inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.update_file(self.__inputs_path)
        # load inputs
        else:
            print('Picke-loading data...')
            with open(self.__inputs_path, 'rb') as handle:
                self.__inputs: List[Dict[str, torch.Tensor]] = pickle.load(handle)

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
