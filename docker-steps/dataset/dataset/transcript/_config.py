from dataset import MyDatasetConfig
from tokenizer import MyDNATokenizer


class TranscriptDatasetConfig(MyDatasetConfig):
    def __init__(
            self,
            genes_panel_path: str,
            transcript_dir: str,
            len_read: int = 150,
            len_kmer: int = 6,
            n_words: int = 30,
            tokenizer: MyDNATokenizer = None,
            **kwargs
    ):
        # check if tokenizer is None
        assert tokenizer is not None
        # call super class
        super().__init__(
            hyper_parameters={
                'genes_panel_path': genes_panel_path,
                'transcript_dir': transcript_dir,
                'len_read': len_read,
                'len_kmer': len_kmer,
                'n_words': n_words,
                'tokenizer': tokenizer,
            },
            **kwargs
        )

    @property
    def genes_panel_path(self) -> str:
        return self.hyper_parameters['genes_panel_path']

    @property
    def transcript_dir(self) -> str:
        return self.hyper_parameters['transcript_dir']

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
