from typing import Union
from typing import Tuple
from typing import List
from typing import Dict

from tqdm import tqdm
import pandas as pd
import torch
import os

from Bio import SeqIO

from tokenizer import MyDNATokenizer


def split_reads_file_on_processes(
        reads_files: List[str],
        n_proc: int
) -> List[List[str]]:
    # get number of files and number of files for each process
    n_files: int = len(reads_files)
    n_files_for_process: int = n_files // n_proc
    rest: int = n_files % n_proc
    # split files on different process
    fasta_files_for_each_process: List[List[str]] = []
    rest_added: int = 0
    for i in range(n_proc):
        start: int = i * n_files_for_process + rest_added
        if rest > i:
            end: int = start + n_files_for_process + 1
            fasta_files_for_each_process.append(reads_files[start:end])
            rest_added += 1
        else:
            end: int = start + n_files_for_process
            fasta_files_for_each_process.append(reads_files[start:end])

    return fasta_files_for_each_process


def split_dataset_on_processes(
        dataset: pd.DataFrame,
        n_proc: int
) -> List[Tuple[int, int]]:
    # get number of rows for each process
    n_rows: int = len(dataset)
    n_rows_for_process: int = n_rows // n_proc
    rest: int = n_rows % n_proc
    # split files on different process
    rows_for_each_process: List[(int, int)] = []
    rest_added: int = 0
    for i in range(n_proc):
        start: int = i * n_rows_for_process + rest_added
        if rest > i:
            end: int = start + n_rows_for_process + 1
            rows_for_each_process.append((start, end))
            rest_added += 1
        else:
            end: int = start + n_rows_for_process
            rows_for_each_process.append((start, end))

    return rows_for_each_process


def generate_kmers_from_sequences(
        reads_files: List[str],
        dir_path: str,
        len_kmer: int,
        labels: Dict[str, int]
) -> pd.DataFrame:
    # init dataset
    dataset: pd.DataFrame = pd.DataFrame()
    # for each read file
    for reads_file in tqdm(reads_files, total=len(reads_files), desc='Generating kmers...'):
        # open file with SeqIO
        fasta_file = SeqIO.parse(open(
            os.path.join(dir_path, f'{reads_file}.reads')
        ), 'fasta')
        # get kmers of all read of file
        for reads in fasta_file:
            sequence: str = reads.seq
            columns: List[str] = []
            values: List[Union[str, int]] = []
            n_kmers: int = len(sequence) + 1 - len_kmer
            for i in range(n_kmers):
                columns.append(f'k_{i}')
                values.append(sequence[i:i + len_kmer].__str__())
            # append label value
            values.append(labels[reads_file])
            columns.append('label')
            # create row by values
            row_dataset: pd.DataFrame = pd.DataFrame(
                [
                    values
                ], columns=columns
            )
            # append row on local dataset
            dataset = pd.concat([dataset, row_dataset])

    return dataset


def generate_sentences_from_kmers(
        rows_index: Tuple[int, int],
        dataset: pd.DataFrame,
        n_words: int
) -> pd.DataFrame:
    # init dataset
    sentences_dataset: pd.DataFrame = pd.DataFrame()
    # get start and end indexes
    start, end = rows_index
    for index in tqdm(range(start, end), total=(end - start), desc='Generating sentences...'):
        # get row of dataset with index
        row: pd.DataFrame = dataset.iloc[[index]]
        # drop NaN values
        row = row.dropna(axis='columns')
        # get kmers and label from row
        kmers: List[str] = list(row.values[0][:-1])
        label: int = int(row.values[0][-1])
        # generate sentences
        n_sentences: int = len(kmers) + 1 - n_words
        if n_sentences < 1:
            continue
        for i in range(n_sentences):
            sentence = ' '.join(kmers[i:i + n_words])
            row_dataset: pd.DataFrame = pd.DataFrame(
                [[
                    sentence,
                    label
                ]], columns=['sentence', 'label']
            )
            # append row on local dataset
            sentences_dataset = pd.concat([sentences_dataset, row_dataset])

    return sentences_dataset


def generate_kmers_from_dataset(
        rows_index: Tuple[int, int],
        dataset: pd.DataFrame,
        len_kmer: int
) -> pd.DataFrame:
    # init dataset
    kmers_dataset: pd.DataFrame = pd.DataFrame()
    # get start and end indexes
    start, end = rows_index
    for index in tqdm(range(start, end), total=(end - start), desc='Generating kmers...'):
        # get row of dataset with index
        row: pd.DataFrame = dataset.iloc[[index]]
        # get read and number of kmers
        read: str = row.values[0][0]
        n_kmers: int = len(read) + 1 - len_kmer
        # init values list and columns list
        values: List[Union[str, int]] = []
        columns: List[str] = []
        # generate all possible kmers
        for i in range(n_kmers):
            columns.append(f'k_{i}')
            values.append(read[i:i + len_kmer].__str__())
        # append value remaining
        columns = columns + list(dataset.columns[1:])
        values = values + list(row.values[0][1:])
        # create row by values
        row_dataset: pd.DataFrame = pd.DataFrame(
            [
                values
            ], columns=columns
        )
        # append row on local dataset
        kmers_dataset = pd.concat([kmers_dataset, row_dataset])

    return kmers_dataset


def generate_sentences_encoded_from_dataset(
        rows_index: Tuple[int, int],
        dataset: pd.DataFrame,
        n_words: int,
        n_kmers: int,
        n_sentences: int,
        tokenizer: MyDNATokenizer
) -> List[Dict[str, Union[List[Dict[str, torch.Tensor]], torch.Tensor]]]:
    # init inputs
    inputs: List[Dict[str, Union[List[Dict[str, torch.Tensor]], torch.Tensor]]] = []
    # get start and end indexes
    start, end = rows_index
    for index in tqdm(range(start, end), total=(end - start), desc='Generating sentences encoded...'):
        # get row of dataset with index
        row: pd.DataFrame = dataset.iloc[[index]]
        # get all kmers in this row
        kmers: List[str] = row.values[0][:n_kmers]
        # get all sentences of n_words
        sentences: List[str] = [' '.join(kmers[i:i + n_words]) for i in range(n_sentences)]
        # tokenize all sentences
        sentences_tokenized: List[Dict[str, List[int]]] = [
            tokenizer.encode_plus(
                sentence,
                padding='max-length',
                add_special_tokens=True,
                truncation=True,
                max_length=n_words + 2
            ) for sentence in sentences
        ]
        # extract tensor from sentences tokenized
        matrix_input_ids: List[torch.Tensor] = []
        matrix_attention_mask: List[torch.Tensor] = []
        matrix_token_type_ids: List[torch.Tensor] = []
        for sentence_tokenized in sentences_tokenized:
            input_ids: List[int] = sentence_tokenized['input_ids']
            token_type_ids: List[int] = sentence_tokenized['token_type_ids']
            attention_mask: List[int] = sentence_tokenized['attention_mask']
            # zero-pad up to the sequence length.
            padding_length: int = n_words + 2 - len(input_ids)
            input_ids: List[int] = input_ids + ([0] * padding_length)
            attention_mask: List[int] = attention_mask + ([1] * padding_length)
            token_type_ids: List[int] = token_type_ids + ([0] * padding_length)
            # append read_input
            matrix_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            matrix_attention_mask.append(torch.tensor(attention_mask, dtype=torch.int))
            matrix_token_type_ids.append(torch.tensor(token_type_ids, dtype=torch.int))

        # append read_inputs to inputs
        inputs.append({
            'matrix_input_ids': torch.stack(matrix_input_ids),
            'matrix_attention_mask': torch.stack(matrix_attention_mask),
            'matrix_token_type_ids': torch.stack(matrix_token_type_ids),
            'label': torch.tensor([row.values[0][-1]], dtype=torch.long)
        })

    return inputs


def encode_sentences(
        rows_index: Tuple[int, int],
        dataset: pd.DataFrame,
        n_words: int,
        tokenizer: MyDNATokenizer
) -> List[Dict[str, torch.Tensor]]:
    # init inputs
    inputs: List[Dict[str, torch.Tensor]] = []
    # get start and end indexes
    start, end = rows_index
    # tokenizing sequences
    for index in tqdm(range(start, end), total=(end - start), desc='Tokenization of sentences...'):
        # extract sentence by index of row
        row: pd.DataFrame = dataset.iloc[[index]]
        sentence: str = row.values[0][0]
        sentence_tokenized = tokenizer.encode_plus(
            sentence,
            padding='max-length',
            add_special_tokens=True,
            truncation=True,
            max_length=n_words + 2
        )
        input_ids: List[int] = sentence_tokenized['input_ids']
        token_type_ids: List[int] = sentence_tokenized['token_type_ids']
        attention_mask: List[int] = sentence_tokenized['attention_mask']
        # zero-pad up to the sequence length.
        padding_length: int = n_words + 2 - len(input_ids)
        input_ids: List[int] = input_ids + ([0] * padding_length)
        attention_mask: List[int] = attention_mask + ([1] * padding_length)
        token_type_ids: List[int] = token_type_ids + ([0] * padding_length)
        # append sentence input to local inputs
        inputs.append(
            {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.int),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.int),
                'label': torch.tensor([row.values[0][1]], dtype=torch.long)
            }
        )

    return inputs
