from dataset import MyDatasetConfig
from tokenizer import MyDNATokenizer


class FusionDatasetConfig(MyDatasetConfig):
    def __init__(
            self,
            genes_panel_path: str,
            len_read: int = 150,
            len_kmer: int = 6,
            n_words: int = 30,
            tokenizer: MyDNATokenizer = None,
            n_fusion: int = 30,
            **kwargs
    ):
        # check if tokenizer is None
        assert tokenizer is not None
        # call super class
        super().__init__(
            hyper_parameters={
                'genes_panel_path': genes_panel_path,
                'len_read': len_read,
                'len_kmer': len_kmer,
                'n_words': n_words,
                'tokenizer': tokenizer,
                'n_fusion': n_fusion,
            },
            **kwargs
        )

    @property
    def genes_panel_path(self) -> str:
        return self.hyper_parameters['genes_panel_path']

    @property
    def len_read(self) -> int:
        return self.hyper_parameters['len_read']

    @property
    def len_kmer(self) -> int:
        return self.hyper_parameters['len_kmer']

    @property
    def n_words(self) -> int:
        return self.hyper_parameters['n_words']

    @property
    def tokenizer(self) -> MyDNATokenizer:
        return self.hyper_parameters['tokenizer']

    @property
    def n_fusion(self) -> int:
        return self.hyper_parameters['n_fusion']
