from typing import List

from itertools import product
import os

from tokenizer import MyDNATokenizer


class DNABertTokenizer(MyDNATokenizer):
    def __init__(
            self,
            root_dir: str,
            len_kmer: int = 6,
            add_n: bool = True,
            do_lower_case: bool = False,
            pad_token: str = '[PAD]',
            unk_token: str = '[UNK]',
            cls_token: str = '[CLS]',
            sep_token: str = '[SEP]',
            mask_token: str = '[MASK]'
    ):
        # define path
        self.__vocab_name = f'kmer_{len_kmer}{"_n" if add_n else ""}'
        self.__vocab_path = os.path.join(root_dir, f'{self.__vocab_name}.txt')

        # check if vocab is defined
        if not os.path.exists(self.__vocab_path):
            # create vocab
            vocabs: List[str] = []
            words: List[str] = ['A', 'T', 'C', 'G', 'N']
            if not add_n:
                words = words[:-1]
            for comb in product(words, repeat=len_kmer):
                vocabs.append(''.join(comb))
            with open(self.__vocab_path, "w") as vocab_file:
                for special_token in [pad_token, unk_token, cls_token, sep_token, mask_token]:
                    vocab_file.write(f'{special_token}\n')
                for vocab in vocabs:
                    vocab_file.write(f'{vocab}\n')

        super().__init__(
            vocab_name=self.__vocab_name,
            vocab_file=self.__vocab_path,
            len_kmer=len_kmer,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token
        )

    def save_vocabulary(self, save_directory):
        raise NotImplementedError
