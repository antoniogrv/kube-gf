from typing import List
from typing import Dict
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import logging
import os

from sklearn.preprocessing import label_binarize

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def save_result(
        result_csv_path: str,
        len_read: int,
        len_kmer: int,
        n_words: int,
        tokenizer_selected: str,
        hyper_parameters: Dict[str, Any],
        y_true: np.ndarray,
        y_pred: np.ndarray
):
    # init columns of result df
    columns = ['len_read', 'len_kmer', 'n_words', 'tokenizer_selected']
    columns += list(hyper_parameters.keys())
    columns += ['accuracy', 'precision', 'recall', 'f1-score']

    # create row of df
    values = [len_read, len_kmer, n_words, tokenizer_selected]
    values += [hyper_parameters[p] for p in hyper_parameters.keys()]
    accuracy = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='weighted',
        zero_division=1
    )
    values += [accuracy, precision, recall, f_score]
    result_csv: pd.DataFrame = pd.DataFrame(
        [
            values
        ],
        columns=columns
    )

    # check if result dataset exists
    if os.path.exists(result_csv_path):
        global_results_csv: pd.DataFrame = pd.read_csv(result_csv_path)
        global_results_csv = pd.concat([global_results_csv, result_csv])
        global_results_csv = global_results_csv.sort_values(by=['accuracy'], ascending=False)
        global_results_csv.to_csv(result_csv_path, index=False)
    else:
        result_csv.to_csv(result_csv_path, index=False)


def plot_confusion_matrix(
        cm,
        target_names: List[str],
        confusion_matrix_path: str,
        title: str = 'Confusion matrix',
        cmap=None,
        normalize: bool = True
):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    miss_class = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, miss_class))
    plt.savefig(confusion_matrix_path)
    plt.close()


def plot_roc_curve(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        target_names: List[str],
        roc_curves_path: str
):
    # init dicts
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # init fig
    plt.figure(figsize=(10, 10))

    if len(target_names) == 2:
        fpr[0], tpr[0], _ = roc_curve(y_true, y_probs)
        roc_auc[0] = auc(fpr[0], tpr[0])
        # plot roc curve
        plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % (roc_auc[0]))
    else:
        # binarize vector of true
        y_true_binary = label_binarize(y_true, classes=list(range(len(target_names))))
        # evaluate ROC and auc for each class
        for i in range(len(target_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # evaluate micro-averaged ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # plot roc curves
        for idx, class_name in enumerate(target_names):
            plt.plot(fpr[idx], tpr[idx], label='ROC curve (area = %0.2f) - %s' % (roc_auc[idx], class_name))

    # options of plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.savefig(roc_curves_path)
    plt.close()


def log_results(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        target_names: List[str],
        logger: logging.Logger,
        test_dir: str
) -> np.ndarray:
    # evaluate y_pred
    if len(target_names) == 2:
        y_pred: np.ndarray = y_probs.round()
    else:
        y_pred: np.ndarray = np.argmax(y_probs, axis=1)
    # get accuracy and balanced accuracy
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    # get precision, recall and f_score
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='weighted',
        zero_division=1
    )
    logger.info(f'Accuracy score: {accuracy}')
    logger.info(f'Balanced accuracy score: {balanced_accuracy}')
    logger.info(f'Weighted precision: {precision}')
    logger.info(f'Weighted recall: {recall}')
    logger.info(f'Weighted f_score: {f_score}')
    logger.info('\n')

    # get classification report
    report: str = classification_report(
        y_true,
        y_pred,
        digits=3,
        zero_division=1,
        target_names=target_names
    )
    logger.info(report)

    # plot confusion matrix
    plot_confusion_matrix(
        cm=confusion_matrix(y_true, y_pred, normalize='true'),
        target_names=target_names,
        confusion_matrix_path=os.path.join(test_dir, 'confusion_matrix.svg')
    )

    # plot roc curves
    plot_roc_curve(
        y_true=y_true,
        y_probs=y_probs,
        target_names=target_names,
        roc_curves_path=os.path.join(test_dir, 'roc_curves.svg')
    )

    return y_pred
