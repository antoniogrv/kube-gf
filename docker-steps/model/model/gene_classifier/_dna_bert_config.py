from model import MyModelConfig


class GCDNABertModelConfig(MyModelConfig):
    def __init__(
            self,
            vocab_size: int = 15630,
            hidden_size: int = 1024,
            n_hidden_layers: int = 1,
            dropout: float = 0.6,
            n_attention_heads: int = 2,
            **kwargs
    ):
        # call super class
        super().__init__(
            hyper_parameters={
                'vocab_size': vocab_size,
                'hidden_size': hidden_size,
                'n_hidden_layers': n_hidden_layers,
                'dropout': dropout,
                'n_attention_heads': n_attention_heads
            },
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return self.hyper_parameters['vocab_size']

    @property
    def hidden_size(self) -> int:
        return self.hyper_parameters['hidden_size']

    @property
    def n_hidden_layers(self) -> int:
        return self.hyper_parameters['n_hidden_layers']

    @property
    def dropout(self) -> float:
        return self.hyper_parameters['dropout']

    @property
    def n_attention_heads(self) -> int:
        return self.hyper_parameters['n_attention_heads']
