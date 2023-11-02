from typing import Optional
from typing import Tuple
from typing import Dict

from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn

from model import MyModel
from model import MyModelConfig


class GeneClassifier(MyModel, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
            self,
            model_dir: str,
            model_name: str,
            config: MyModelConfig,
            n_classes: int = 1,
            weights: Optional[torch.Tensor] = None
    ):
        super().__init__(
            model_dir=model_dir,
            model_name=model_name,
            config=config,
            n_classes=n_classes,
            weights=weights
        )

    @abstractmethod
    def load_data(
            self,
            batch,
            device: torch.device
    ) -> Tuple[Dict[str, any], torch.Tensor]:
        pass

    @abstractmethod
    def step(
            self,
            inputs: Dict[str, any]
    ) -> any:
        pass

    @abstractmethod
    def compute_loss(
            self,
            target: torch.Tensor,
            output: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def embedding_step(
            self,
            inputs: Dict[str, any]
    ) -> any:
        pass

    @abstractmethod
    def get_embedding_layer(self) -> nn.Module:
        pass

