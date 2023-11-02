from typing import Optional
from typing import Tuple
from typing import Final
from typing import Dict

from dotenv import load_dotenv
import numpy as np
import logging
import torch
import os

from tokenizer import MyDNATokenizer
from tokenizer import DNABertTokenizer

from dataset import TranscriptDatasetConfig
from dataset import TranscriptDataset
from torch.utils.data import DataLoader

from model import MyModelConfig
from model import GeneClassifier
from model import evaluate_weights

from model import GCDNABertModelConfig
from model import GCDNABert

from torch.optim import AdamW

from utils import SEPARATOR
from utils import define_gene_classifier_inputs
from utils import create_test_id
from utils import init_test
from utils import setup_logger
from utils import log_results
from utils import save_result
from utils import close_loggers


def entrypoint(
        test_csv_path: str,
        train_csv_path: str,
        val_csv_path: str,
        model_path: str,
        results_path: str,
        len_read: int,
        len_kmer: int,
        n_words: int,
        tokenizer_selected: str,
        model_selected: str,
        hyper_parameters: Dict[str, any],
        batch_size: int,
        re_train: bool,
        grid_search: bool,
) -> Tuple[str, MyModelConfig]:
    
    # get value from .env
    root_dir: Final = 'data'

    # init tokenizer
    tokenizer: Optional[MyDNATokenizer] = None
    if tokenizer_selected == 'dna_bert':
        tokenizer = DNABertTokenizer(
            root_dir=root_dir,
            len_kmer=len_kmer,
            add_n=False
        )
    elif tokenizer_selected == 'dna_bert_n':
        tokenizer = DNABertTokenizer(
            root_dir=root_dir,
            len_kmer=len_kmer,
            add_n=True
        )

    # init configuration
    model_config: Optional[MyModelConfig] = None
    if model_selected == 'dna_bert':
        model_config = GCDNABertModelConfig(
            vocab_size=tokenizer.vocab_size,
            **hyper_parameters
        )

    # generate test id
    test_id: str = "tmp"

    # create dataset configuration
    dataset_conf: TranscriptDatasetConfig = TranscriptDatasetConfig(
        genes_panel_path='data/genes_panel.txt',
        transcript_dir='data/transcripts',
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer=tokenizer
    )

    # get global variables
    task: str = 'gene_classifier'
    result_dir: str = os.path.join(os.getcwd(), 'results')
    model_name: str = 'model'

    # init test
    parent_dir, test_dir, log_dir, model_dir = init_test(
        result_dir=result_dir,
        task=task,
        model_selected=model_selected,
        test_id=test_id,
        model_name=model_name,
        re_train=re_train
    )

    # if the model has not yet been trained
    if not os.path.exists(model_path):
        print('Model not present. Proceeding with training.')

        model_save_dir = os.path.dirname(model_path)
        if not os.path.isdir(model_save_dir): os.makedirs(model_save_dir)

        # init loggers
        logger: logging.Logger = setup_logger(
            'logger',
            os.path.join(log_dir, 'logger.log')
        )
        train_logger: logging.Logger = setup_logger(
            'train',
            os.path.join(log_dir, 'train.log')
        )

        # load train and validation dataset
        print('Preparing Train Dataset for consumption...')
        train_dataset = TranscriptDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='train',
            train_csv_path=train_csv_path,
            test_csv_path="",
            val_csv_path=val_csv_path,
        )
        print('Preparing Validation Dataset for consumption...')
        val_dataset = TranscriptDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='val',
            val_csv_path=val_csv_path,
            train_csv_path=train_csv_path,
            test_csv_path="",
        )

        # log information
        logger.info(f'Read len: {len_read}')
        logger.info(f'Kmers len: {len_kmer}')
        logger.info(f'Words inside a sentence: {n_words}')
        logger.info(f'Tokenizer used: {tokenizer_selected}')
        logger.info(SEPARATOR)
        logger.info(f'Number of train sentences: {len(train_dataset)}')
        logger.info(f'Number of val sentences: {len(val_dataset)}')
        logger.info(f'Number of class: {train_dataset.classes()}')
        logger.info(f'Batch size: {batch_size}')
        logger.info(SEPARATOR)
        logger.info(f'Number of train sentences: {len(train_dataset)}')
        logger.info(f'Number of val sentences: {len(val_dataset)}')
        logger.info(f'Number of class: {train_dataset.classes()}')
        logger.info(f'Batch size: {batch_size}')
        logger.info(SEPARATOR)
        # print dataset status
        logger.info('No. records train set')
        logger.info(train_dataset.print_dataset_status())
        logger.info('No. records val set')
        logger.info(val_dataset.print_dataset_status())

        # load train and validation dataloader+
        print('Preparing Train Dataloader...')
        train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        print('Preparing Validation Dataloader...')
        val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # set device gpu if cuda is available
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('CUDA: ', torch.cuda.is_available())
        print('Device: ', device)
        
        # evaluating weights for criterion function
        y_true = []
        for idx, label in enumerate(train_dataset.get_dataset_status()):
            y_true = np.append(y_true, [idx] * label)
        class_weights: torch.Tensor = evaluate_weights(
            y_true=y_true,
            binary=train_dataset.classes() == 2
        ).to(device)

        # define model
        model: GeneClassifier = GCDNABert(
            model_dir=model_dir,
            model_name=model_name,
            config=model_config,
            n_classes=train_dataset.classes(),
            weights=class_weights
        )

        # log model hyper parameters
        logger.info('Model hyper parameters')
        logger.info(model_config)

        # init optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=5e-5,
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        # put model on device available
        model.to(device)
        # train it
        model.train_model(
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epochs=1,
            evaluation=True,
            val_loader=val_loader,
            logger=train_logger
        )

        # close loggers
        close_loggers([train_logger, logger])
        del train_logger
        del logger

    # if the model is already trained and the grid search parameter is set to true then stop
    else:
        print('Model is present. Proceeding with testing.')

        results_save_dir = os.path.dirname(results_path)
        if not os.path.isdir(results_save_dir): os.makedirs(results_save_dir)

        # init loggers
        logger: logging.Logger = setup_logger(
            'logger',
            os.path.join(log_dir, 'logger.log')
        )
        result_logger: logging.Logger = setup_logger(
            'result',
            os.path.join(log_dir, 'result.log')
        )

        print('Preparing test dataset for consumption...')
        test_dataset = TranscriptDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='test',
            train_csv_path="",
            val_csv_path="",
            test_csv_path=test_csv_path,
        )

        # log test dataset status
        logger.info('No. records test set')
        logger.info(test_dataset.print_dataset_status())

        # load model
        model: GeneClassifier = torch.load(model_path)
        # set device gpu if cuda is available
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # set model on gpu
        model.to(device)

        # create test data loader
        print('Preparing test dataloader...')
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # test model
        print('Predicting...')
        y_true, y_probs = model.predict(
            test_loader=test_loader,
            device=device
        )

        # log results
        print('Logging results...')
        y_pred = log_results(
            y_true=y_true,
            y_probs=y_probs,
            target_names=list(test_dataset.get_labels_dict().keys()),
            logger=result_logger,
            test_dir=test_dir
        )

        # close loggers
        close_loggers([logger, result_logger])
        del logger
        del result_logger

        # save result
        print('Saving results...')
        save_result(
            result_csv_path=results_path,
            len_read=len_read,
            len_kmer=len_kmer,
            n_words=n_words,
            tokenizer_selected=tokenizer_selected,
            hyper_parameters=model_config.hyper_parameters,
            y_true=y_true,
            y_pred=y_pred
        )

    # return model_path
    return model_path, model_config


if __name__ == '__main__':
    # define inputs for this script
    __args, __hyper_parameters = define_gene_classifier_inputs()

    # execute train_gene_classifier method
    entrypoint(
        **__args,
        hyper_parameters=__hyper_parameters
    )
