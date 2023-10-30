from typing import Dict

from abc import ABCMeta
from abc import abstractmethod

from checksumdir import dirhash
import pandas as pd
import hashlib
import pickle
import os

from dataset import MyDatasetConfig
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset, metaclass=ABCMeta):
    def __init__(
            self,
            root_dir: str,
            check_dir_name: str,
            check_dict_name: str,
            conf: MyDatasetConfig,
            dataset_type: str
    ):
        super().__init__()
        self.__root_dir: str = root_dir
        self.__conf: MyDatasetConfig = conf

        assert dataset_type in ['train', 'val', 'test']
        self.__dataset_type: str = dataset_type

        # if processed dir doesn't exist, create it otherwise
        self.__processed_dir = os.path.join(self.__root_dir, 'processed_dir')
        if not os.path.exists(self.__processed_dir):
            os.makedirs(self.__processed_dir)

        # if check dir doesn't exist, create it otherwise
        self.__check_dir = os.path.join(self.__root_dir, check_dir_name)
        if not os.path.exists(self.__check_dir):
            os.makedirs(self.__check_dir)
        # save check dict path
        self.__check_dict_path = os.path.join(self.__check_dir, f'{check_dict_name}.pkl')

        # load check dict if exists, create it otherwise
        if os.path.exists(self.__check_dict_path):
            with open(self.__check_dict_path, 'rb') as handle:
                self.__check_dict: Dict[str, str] = pickle.load(handle)
        else:
            self.__check_dict: Dict[str, str] = {}

        # # if inputs dir doesn't exist, create it otherwise
        self.__inputs_dir = os.path.join(self.__root_dir, 'inputs_model')
        if not os.path.exists(self.__inputs_dir):
            os.makedirs(self.__inputs_dir)

    def check_dir(self, dir_path: str) -> bool:
        if not os.path.exists(dir_path):
            return False
        if dir_path in self.__check_dict:
            if dirhash(dir_path, 'sha256') == self.__check_dict[dir_path]:
                return True
            else:
                return False
        else:
            return False

    def update_dir(self, dir_path: str) -> None:
        self.__check_dict[dir_path] = dirhash(dir_path, 'sha256')
        with open(self.__check_dict_path, 'wb') as handle:
            pickle.dump(self.__check_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def check_dataset(self, dataset_path: str) -> bool:
        if not os.path.exists(dataset_path):
            return False
        if dataset_path in self.__check_dict:
            df: pd.DataFrame = pd.read_csv(dataset_path)
            if hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest() == self.__check_dict[dataset_path]:
                return True
            else:
                return False
        else:
            return False

    def update_dataset(self, dataset_path: str) -> None:
        df: pd.DataFrame = pd.read_csv(dataset_path)
        self.__check_dict[dataset_path] = hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()
        with open(self.__check_dict_path, 'wb') as handle:
            pickle.dump(self.__check_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def check_file(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        if file_path in self.__check_dict:
            with open(file_path, 'rb') as handle:
                if hashlib.sha1(handle.read()).hexdigest() == self.__check_dict[file_path]:
                    return True
                else:
                    return False
        else:
            return False

    def update_file(self, file_path: str) -> None:
        with open(file_path, 'rb') as handle:
            self.__check_dict[file_path] = hashlib.sha1(handle.read()).hexdigest()
        with open(self.__check_dict_path, 'wb') as handle:
            pickle.dump(self.__check_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def root_dir(self) -> str:
        return self.__root_dir

    @property
    def processed_dir(self) -> str:
        return self.__processed_dir

    @property
    def inputs_dir(self) -> str:
        return self.__inputs_dir

    @property
    def conf(self) -> MyDatasetConfig:
        return self.__conf

    @property
    def dataset_type(self) -> str:
        return self.__dataset_type

    @abstractmethod
    def get_labels_dict(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_dataset_status(self):
        pass

    @abstractmethod
    def print_dataset_status(self) -> str:
        pass
