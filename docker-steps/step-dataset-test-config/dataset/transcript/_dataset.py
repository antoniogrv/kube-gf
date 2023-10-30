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

        # ======================== Load Gene Panel ======================== #
        print('Loading Genes...')
        self.__gene_panel_path: str = conf.genes_panel_path
        with open(self.__gene_panel_path, 'r') as gene_panel_file:
            self.__genes_list: List[str] = gene_panel_file.read().split('\n')
        self.update_file(self.__gene_panel_path)

        # ====================== Labelling genes Step ===================== #
        print('Labelling Genes...')
        self.__labels_path: str = os.path.join(self.inputs_dir, 'transcript_label.pkl')
        # check if labels dict are already generated
        labelling_phase: bool = self.check_file(self.__gene_panel_path) and self.check_file(self.__labels_path)
        if not labelling_phase:
            self.__labels: Dict[str, int] = {}
            for idx, gene in enumerate(self.__genes_list):
                self.__labels[gene] = idx
            with open(self.__labels_path, 'wb') as handle:
                pickle.dump(self.__labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.update_file(self.__labels_path)
        else:
            with open(self.__labels_path, 'rb') as handle:
                self.__labels: Dict[str, int] = pickle.load(handle)

        # ===================== Generation reads Step ==================== #
        print('Generating Reads...')
        __gt_shredder_dir: str = os.path.join(self.root_dir, f'gt_shredder_{conf.len_read}')
        # check if reads are already generated
        generation_reads_phase: bool = labelling_phase and self.check_dir(__gt_shredder_dir)
        if not generation_reads_phase:
            # create directory if it doesn't exist
            if not os.path.exists(__gt_shredder_dir):
                os.makedirs(__gt_shredder_dir)
            # execute gt-shredder on all fastq files
            gt_shredder(
                transcript_dir=conf.transcript_dir,
                output_dir=__gt_shredder_dir,
                len_read=conf.len_read
            )
            # update step info
            self.update_dir(__gt_shredder_dir)

        # ===================== Generation kmers Step =================== #
        print('Generating Kmers...')
        __kmers_dataset_path: str = os.path.join(
            self.processed_dir,
            f'transcript_{conf.len_read}_'
            f'kmer_{conf.len_kmer}.csv'
        )
        # check if kmers dataset is already generated
        generation_kmers_dataset_phase: bool = generation_reads_phase and self.check_dataset(__kmers_dataset_path)
        if not generation_kmers_dataset_phase:
            # split genes on processes
            reads_files_for_each_process: List[List[str]] = split_reads_file_on_processes(
                reads_files=list(self.__labels.keys()),
                n_proc=os.cpu_count()
            )
            # init dataset
            __kmers_dataset: pd.DataFrame = pd.DataFrame()
            # call generate kmers from sequence on multi processes
            with Pool(os.cpu_count()) as pool:
                results = pool.imap(partial(
                    generate_kmers_from_sequences,
                    dir_path=__gt_shredder_dir,
                    len_kmer=conf.len_kmer,
                    labels=self.__labels
                ), reads_files_for_each_process)
                # append all local dataset to global dataset
                for local_dataset in results:
                    __kmers_dataset = pd.concat([__kmers_dataset, local_dataset])
            # save kmers dataset as csv
            __kmers_dataset.to_csv(__kmers_dataset_path, index=False)
            self.update_dataset(__kmers_dataset_path)

        # ============== Generation of train, val, test Step ============== #
        print('Generating Dataset...')
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
        # check if train, val and test set are already generateds
        generation_sets_phase_flag: bool = generation_kmers_dataset_phase and (
                self.check_dataset(__train_dataset_path) and
                self.check_dataset(__val_dataset_path) and
                self.check_dataset(__test_dataset_path)
        )
        if not generation_sets_phase_flag:
            # load kmers dataset
            __kmers_dataset: pd.DataFrame = pd.read_csv(__kmers_dataset_path)
            # split dataset in train, val and test set
            __train_dataset, __test_dataset = train_test_split(
                __kmers_dataset,
                test_size=0.1
            )
            __train_dataset, __val_dataset = train_test_split(
                __train_dataset,
                test_size=0.11
            )
            # group datasets and dataset paths
            __datasets: List[pd.DataFrame] = [
                __test_dataset
            ]
            __dataset_paths: List[str] = [
                __train_dataset_path,
                __val_dataset_path,
                __test_dataset_path
            ]
            # generate sentences for each list of kmers
            for i in range(1):
                print('Generating Sentences... (', i, ')')
                __datasets[i].reset_index(drop=True, inplace=True)
                # split dataset on processes
                rows_for_each_process: List[Tuple[int, int]] = split_dataset_on_processes(
                    __datasets[i],
                    os.cpu_count()
                )
                # init dataset
                __sentences_dataset: pd.DataFrame = pd.DataFrame()
                # call generate kmers from sequence on multi processes
                with Pool(os.cpu_count()) as pool:
                    results = pool.imap(partial(
                        generate_sentences_from_kmers,
                        dataset=__datasets[i],
                        n_words=conf.n_words
                    ), rows_for_each_process)
                    # append all local dataset to global dataset
                    for local_dataset in results:
                        __sentences_dataset = pd.concat([__sentences_dataset, local_dataset])
                # shuffles rows, saves datasets as csv, and updates hashes
                __sentences_dataset = __sentences_dataset.sample(frac=1).reset_index(drop=True)
                __sentences_dataset.to_csv(__dataset_paths[i], index=False)
                self.update_dataset(__dataset_paths[i])

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
        print('Evaluating Model Inputs...')
        self.__inputs_path: str = os.path.join(
            self.inputs_dir,
            f'transcript_{conf.len_read}_'
            f'kmer_{conf.len_kmer}_'
            f'n_words_{conf.n_words}_'
            f'tokenizer_{conf.tokenizer}_'
            f'{self.dataset_type}.pkl'
        )
        print('Inputs Path: ', self.__inputs_path)
        # check if inputs tensor are already generateds
        generation_inputs_phase: bool = generation_sets_phase_flag and self.check_file(self.__inputs_path)
        if not generation_inputs_phase:
            print('Generating Model Inputs...')
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
                print('Pickle-dumping Data...')
                pickle.dump(self.__inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.update_file(self.__inputs_path)
            print('Done!')
        # load inputs
        else:
            print('Skipping Model Inputs Generation...')
            with open(self.__inputs_path, 'rb') as handle:
                print('Pickle-loading Data...')
                self.__inputs: List[Dict[str, torch.Tensor]] = pickle.load(handle)
                print('Done!')

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
