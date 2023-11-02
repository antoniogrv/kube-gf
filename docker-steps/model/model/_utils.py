import numpy as np
import torch

from sklearn.utils import class_weight


def evaluate_weights(
        y_true: np.ndarray,
        binary: bool = False
) -> torch.Tensor:
    if binary:
        class_weights = (y_true == 0.).sum() / y_true.sum()
    else:
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_true), y=y_true)
    return torch.tensor(class_weights, dtype=torch.float32)
