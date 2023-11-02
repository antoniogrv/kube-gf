from typing import Optional
from typing import Tuple
from typing import Dict

import torch
import torch.nn as nn

from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss

from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_bert import BERT_INPUTS_DOCSTRING
from transformers import (
    BertModel,
    BertConfig
)

from model.gene_classifier import GeneClassifier
from model.gene_classifier import GCDNABertModelConfig


class GCDNABert(GeneClassifier):
    def __init__(
            self,
            model_dir: str,
            model_name: str = 'model',
            config: GCDNABertModelConfig = None,
            n_classes: int = 1,
            weights: Optional[torch.Tensor] = None
    ):
        # call super class
        super().__init__(
            model_dir=model_dir,
            model_name=model_name,
            config=config,
            n_classes=n_classes,
            weights=weights
        )

        # init configuration of model
        __bert_config = BertConfig(
            finetuning_task='dnaprom',
            hidden_act='gelu',
            model_type='bert',
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.n_hidden_layers,
            hidden_dropout_prob=config.dropout,
            num_attention_heads=config.n_attention_heads,
            num_labels=self.n_classes,
        )

        # create model from configuration
        self.bert = BertModel(__bert_config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.n_classes if self.n_classes > 2 else 1)

        # init loss function
        if self.n_classes == 2:
            self.__loss = BCEWithLogitsLoss(pos_weight=weights)
        else:
            self.__loss = CrossEntropyLoss(weight=weights)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):
        # call bert forward
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # extract pooled output
        pooled_output = outputs[1]

        # dropout and linear output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)

        return outputs

    def load_data(
            self,
            batch, device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # prepare input of batch for classifier
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        target = batch['label'].to(device)
        # return Dict
        return {
                   'input_ids': input_ids,
                   'attention_mask': attention_mask,
                   'token_type_ids': token_type_ids
               }, target

    def step(
            self,
            inputs: Dict[str, torch.Tensor]
    ) -> any:
        # call self.forward
        return self(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )

    def embedding_step(
            self,
            inputs: Dict[str, any]
    ) -> any:
        # call self.bert and return pooled results
        return self.dropout(
            self.bert(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids']
            )[1]
        )

    def compute_loss(
            self,
            target: torch.Tensor,
            output: torch.Tensor
    ) -> torch.Tensor:
        if self.n_classes == 2:
            return self.__loss(output.view(-1), target.view(-1))
        else:
            return self.__loss(output.view(-1, self.n_classes), target.view(-1))

    def get_embedding_layer(self) -> nn.Module:
        return self.bert
