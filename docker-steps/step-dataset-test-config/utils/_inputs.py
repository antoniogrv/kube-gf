from typing import Tuple
from typing import List
from typing import Dict

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def define_general_parameters(
        arg_parser: argparse.ArgumentParser,
) -> None:
    arg_parser.add_argument('-len_read', dest='len_read', action='store',
                            type=int, default=150, help='define length of reads')
    arg_parser.add_argument('-len_kmer', dest='len_kmer', action='store',
                            type=int, default=6, help='define length of kmers')
    arg_parser.add_argument('-n_words', dest='n_words', action='store',
                            type=int, default=20, help='number of kmers inside a sentence')
    arg_parser.add_argument('-tokenizer_selected', dest='tokenizer_selected', action='store',
                            type=str, default='dna_bert_n', help='select the tokenizer to be used')
    arg_parser.add_argument('-grid_search', dest='grid_search', action='store', type=str2bool,
                            default=False, help='set true if this script is launching from grid_search script')


def define_gene_training_parameters(
        arg_parser: argparse.ArgumentParser,
        suffix: str = ''
) -> None:
    arg_parser.add_argument(f'-{suffix}model_selected', dest=f'{suffix}model_selected', action='store',
                            type=str, default='dna_bert', help='select the model to be used')
    arg_parser.add_argument(f'-{suffix}batch_size', dest=f'{suffix}batch_size', action='store',
                            type=int, default=512, help='define batch size')
    arg_parser.add_argument(f'-{suffix}re_train', dest=f'{suffix}re_train', action='store', type=str2bool,
                            default=False, help='set true if you wish to retrain the model despite having already '
                                                'tested with these hyperparameters. Obviously, if the model has been '
                                                'trained on a different dataset you need to set this parameter to true')


def define_fusion_training_parameters(
        arg_parser: argparse.ArgumentParser
) -> None:
    arg_parser.add_argument(f'-model_selected', dest=f'model_selected', action='store',
                            type=str, default='fc', help='select the model to be used')
    arg_parser.add_argument(f'-n_fusion', dest=f'n_fusion', action='store',
                            type=int, default=30, help='number of fusions to be generated per gene by fusim')
    arg_parser.add_argument(f'-batch_size', dest=f'batch_size', action='store',
                            type=int, default=512, help='define batch size')
    arg_parser.add_argument(f'-freeze', dest=f'freeze', action='store', type=str2bool, default=True,
                            help='if set to true, the weights of the gene model embedding layer is not updated')
    arg_parser.add_argument(f'-re_train', dest=f're_train', action='store', type=str2bool,
                            default=False, help='set true if you wish to retrain the model despite having already '
                                                'tested with these hyperparameters. Obviously, if the model has been '
                                                'trained on a different dataset you need to set this parameter to true')


def define_gene_classifier_hyperparameters(
        arg_parser: argparse.ArgumentParser,
        prefix: str = ''
) -> int:
    # add all hyperparameters definition
    hyperparameters_inputs: Dict[str, Dict[str, any]] = {
        f'-{prefix}hidden_size': {
            'dest': f'{prefix}hidden_size',
            'action': 'store',
            'type': int,
            'default': 1024,
            'help': 'define number of hidden channels'
        },
        f'-{prefix}n_hidden_layers': {
            'dest': f'{prefix}n_hidden_layers',
            'action': 'store',
            'type': int,
            'default': 7,
            'help': 'define number of hidden layers'
        },
        f'-{prefix}n_attention_heads': {
            'dest': f'{prefix}n_attention_heads',
            'action': 'store',
            'type': int,
            'default': 1,
            'help': 'define number of attention heads'
        },
        f'-{prefix}dropout': {
            'dest': f'{prefix}dropout',
            'action': 'store',
            'type': float,
            'default': 0.6,
            'help': 'define value of dropout probability'
        }
    }
    # add inputs to input_args
    for input_key, input_values in hyperparameters_inputs.items():
        arg_parser.add_argument(
            input_key,
            **input_values
        )
    # return number of model's hyperparameters
    return len(hyperparameters_inputs)


def define_fusion_classifier_hyperparameters(
        arg_parser: argparse.ArgumentParser,
) -> int:
    # add all hyperparameters definition
    hyperparameters_inputs: Dict[str, Dict[str, any]] = {
        f'-hidden_size': {
            'dest': f'hidden_size',
            'action': 'store',
            'type': int,
            'default': 1024,
            'help': 'define number of hidden channels'
        },
        f'-n_hidden_layers': {
            'dest': f'n_hidden_layers',
            'action': 'store',
            'type': int,
            'default': 2,
            'help': 'define number of hidden layers'
        },
        f'-rnn_type': {
            'dest': f'rnn_type',
            'action': 'store',
            'type': str,
            'default': 'lstm',
            'help': 'define type of recurrent layers'
        },
        f'-n_rnn_layers': {
            'dest': f'n_rnn_layers',
            'action': 'store',
            'type': int,
            'default': 1,
            'help': 'define no. of recurrent layers'
        },
        f'-pooling_op': {
            'dest': f'pooling_op',
            'action': 'store',
            'type': str,
            'default': 'flatten',
            'help': "define type of pooling's operation"
        },
        f'-dropout': {
            'dest': f'dropout',
            'action': 'store',
            'type': float,
            'default': 0.6,
            'help': 'define value of dropout probability'
        }
    }
    # add inputs to input_args
    for input_key, input_values in hyperparameters_inputs.items():
        arg_parser.add_argument(
            input_key,
            **input_values
        )
    # return number of model's hyperparameters
    return len(hyperparameters_inputs)


def check_tokenizer(
        args_dict: Dict[str, str],
) -> None:
    # check tokenizer selected
    if args_dict['tokenizer_selected'] not in ['dna_bert', 'dna_bert_n']:
        raise ValueError('select one of these tokenizers: ["dna_bert", "dna_bert_n"]')


def check_gene_classifier_hyperparameters(
        args_dict: Dict[str, str],
        prefix: str = ''
) -> None:
    # check model selected
    if args_dict[f'{prefix}model_selected'] not in ['dna_bert']:
        raise ValueError('select one of these models: ["dna_bert"]')


def check_fusion_classifier_hyperparameters(
        args_dict: Dict[str, str],
) -> None:
    # check model selected
    if args_dict[f'model_selected'] not in ['fc', 'rnn']:
        raise ValueError('select one of these models: ["fc", "rnn"]')


def define_gene_classifier_inputs() -> Tuple[Dict[str, any], Dict[str, any]]:
    # init parser
    arg_parser = argparse.ArgumentParser()
    # add definition of inputs
    define_general_parameters(arg_parser)
    define_gene_training_parameters(arg_parser, '')
    n_hyperparameters: int = define_gene_classifier_hyperparameters(arg_parser, '')
    # get dict of arguments
    args = arg_parser.parse_args()
    # check value of inputs
    check_tokenizer(vars(args))
    check_gene_classifier_hyperparameters(vars(args), '')

    # split in general and model arguments
    __args: Dict[str, any] = dict(list(vars(args).items())[:-n_hyperparameters])
    __hyperparameters: Dict[str, any] = dict(list(vars(args).items())[n_hyperparameters + 1:])

    return __args, __hyperparameters


def define_fusion_classifier_inputs() -> Tuple[Dict[str, any], Dict[str, any], Dict[str, any]]:
    # init parser
    arg_parser = argparse.ArgumentParser()
    # add definition of inputs
    define_general_parameters(arg_parser)
    define_gene_training_parameters(arg_parser, 'gc_')
    define_fusion_training_parameters(arg_parser)
    n_hyperparameters_gc: int = define_gene_classifier_hyperparameters(arg_parser, 'gc_')
    n_hyperparameters: int = define_fusion_classifier_hyperparameters(arg_parser)
    # get dict of arguments
    args = arg_parser.parse_args()
    # check value of inputs
    check_tokenizer(vars(args))
    check_gene_classifier_hyperparameters(vars(args), 'gc_')
    check_fusion_classifier_hyperparameters(vars(args))
    # split in general and model arguments
    n_args = len(list(vars(args).items()))
    general_args = n_args - n_hyperparameters_gc - n_hyperparameters
    __args: Dict[str, any] = dict(list(vars(args).items())[:general_args])
    __gc_hyperparameters: Dict[str, any] = dict(
        list(vars(args).items())[general_args:general_args + n_hyperparameters_gc])
    __hyperparameters: Dict[str, any] = dict(list(vars(args).items())[-n_hyperparameters:])
    # change prefix of keys in gc dict
    keys: List[str] = list(__gc_hyperparameters.copy().keys())
    for key in keys:
        new_key: str = key[3:]
        __gc_hyperparameters[new_key] = __gc_hyperparameters.pop(key)

    return __args, __gc_hyperparameters, __hyperparameters
