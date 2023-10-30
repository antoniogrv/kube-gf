from typing import Dict
from typing import List

from abc import ABCMeta
from abc import abstractmethod

from tabulate import tabulate


class MyDatasetConfig(metaclass=ABCMeta):
    @abstractmethod
    def __init__(
            self,
            hyper_parameters: Dict[str, any],
            **kwargs
    ):
        super().__init__()
        self.__hyper_parameters: Dict[str, any] = hyper_parameters

    @property
    def hyper_parameters(self) -> Dict[str, any]:
        return self.__hyper_parameters

    def __str__(self) -> str:
        table: List[List[str, any]] = [[parameter, value] for parameter, value in self.__hyper_parameters.items()]
        table_str: str = tabulate(
            tabular_data=table,
            headers=['hyper parameter', 'value'],
            tablefmt='psql',
            numalign='left'
        )
        return f'\n{table_str}\n'
