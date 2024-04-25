#!/Anaconda3/envs/ccjpython/python3
# Created on 2021/6/26 23:11
# Author:tyty
import numpy as np
from math import sqrt
from typing import List, Union
from typing import Callable, List, Union
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss,matthews_corrcoef,confusion_matrix,f1_score
from scipy.stats import spearmanr, pearsonr
# import dtreeviz

def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)

def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return np.sqrt(mean_squared_error(targets, preds, multioutput='raw_values'))

def mse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    return mean_squared_error(targets, preds, multioutput='raw_values')

def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.
    Alternatively, compute accuracy for a multiclass prediction task by picking the largest probability. 

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    if type(preds[0]) == list: # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds] # binary prediction
    return accuracy_score(targets, hard_preds)

def mcc(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.
    Alternatively, compute accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    if type(preds[0]) == list: # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds] # binary prediction
    return matthews_corrcoef(targets, hard_preds)

def conf_mat(targets: List[int], preds: List[float], threshold: float = 0.5):
    if type(preds[0]) == list: # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds] # binary prediction
    return confusion_matrix(targets, hard_preds)

def precision(targets,preds) -> float:
    mat=conf_mat(targets,preds)
    try:
        tn, tp, fp, fn = mat[0, 0], mat[1, 1], mat[0, 1], mat[1, 0]
    except IndexError:
        tn, tp, fp, fn = mat[0, 0], 0, 0, 0
    return tp/(tp+fp)

def specificity(targets,preds) -> float:
    mat=conf_mat(targets,preds)
    try:
        tn, tp, fp, fn = mat[0, 0], mat[1, 1], mat[0, 1], mat[1, 0]
    except IndexError:
        tn, tp, fp, fn = mat[0, 0], 0, 0, 0
    return tn / (tn + fp)

def negPredValue(targets, preds):
    mat=conf_mat(targets,preds)
    try:
        tn, tp, fp, fn = mat[0, 0], mat[1, 1], mat[0, 1], mat[1, 0]
    except IndexError:
        tn, tp, fp, fn = mat[0, 0], 0, 0, 0
    return tn / (tn + fn)


def recall(targets,preds) -> float:
    mat=conf_mat(targets,preds)
    try:
        tn, tp, fp, fn = mat[0, 0], mat[1, 1], mat[0, 1], mat[1, 0]
    except IndexError:
        tn, tp, fp, fn = mat[0, 0], 0, 0, 0
    return tp/(tp+fn)

def ccr(targets, preds) -> float:
    mat = conf_mat(targets, preds)
    try:
        tn, tp, fp, fn = mat[0, 0], mat[1, 1], mat[0, 1], mat[1, 0]
    except IndexError:
        tn, tp, fp, fn = mat[0, 0], 0, 0, 0
    sen = tp * 1.0 / (tp + fn)
    spe = tn * 1.0 / (tn + fp)
    return (sen+spe)/2

def gmeans(targets,preds) -> float:
    mat=conf_mat(targets,preds)
    try:
        tn, tp, fp, fn = mat[0, 0], mat[1, 1], mat[0, 1], mat[1, 0]
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sqrt(recall * specificity)
    except IndexError:
        tn, tp, fp, fn = mat[0,0],0,0,0
        return float('nan')
    
def f1(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    if type(preds[0]) == list: # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds] # binary prediction
    return f1_score(targets,hard_preds)

def fpr(targets, preds):
    mat=conf_mat(targets,preds)
    try:
        tn, tp, fp, fn = mat[0, 0], mat[1, 1], mat[0, 1], mat[1, 0]
    except IndexError:
        tn, tp, fp, fn = mat[0, 0], 0, 0, 0
    return fp/(fp+tn)


def get_norm_metric(metric: str, targets, preds, norm_scales):
    if norm_scales:
        targets =targets/norm_scales
        preds = preds/norm_scales
    return get_metric(metric, targets, preds)

def get_metric(metric: str, targets, preds) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'auc':
        dim =targets.shape[1]
        return np.array([roc_auc_score(targets.reshape(-1,dim)[:,i], preds.reshape(-1,dim)[:,i]) for i in range(dim)])
    
    if metric == 'prc-auc':
        dim =targets.shape[1]
        return np.array([prc_auc(targets.reshape(-1,dim)[:,i], preds.reshape(-1,dim)[:,i]) for i in range(dim)])
    
    if metric == 'rmse':
        return rmse(targets, preds)    
    
    if metric =='mse':
        return mse(targets, preds)
    
    if metric == 'mae':
        return mean_absolute_error(targets, preds, multioutput='raw_values')
    
    if metric == 'r2':
        return r2_score(targets, preds, multioutput='raw_values')
    
    if metric == 'pearsonr':
        dim =targets.shape[1]
        return np.array([pearsonr(targets.reshape(-1,dim)[:,i], preds.reshape(-1,dim)[:,i])[0] for i in range(dim)])
    
    if metric == 'spearmanr':
        dim =targets.shape[1]
        return np.array([spearmanr(targets.reshape(-1,dim)[:,i], preds.reshape(-1,dim)[:,i])[0] for i in range(dim)])
    
    if metric == 'acc':
        dim =targets.shape[1]
        return np.array([accuracy(targets.reshape(-1,dim)[:,i], preds.reshape(-1,dim)[:,i]) for i in range(dim)])
    
    if metric == 'cross_entropy':
        return log_loss(targets, preds)
    
    if metric == 'mcc':
        dim =targets.shape[1]
        return np.array([mcc(targets.reshape(-1,dim)[:,i], preds.reshape(-1,dim)[:,i]) for i in range(dim)])
    
    if metric in ['recall', 'sen', 'tpr']:
        return recall(targets, preds)

    if metric in ['fpr']:
        return fpr(targets, preds)
    
    
    if metric in ['precision', 'ppv']:
        return precision(targets, preds)
    
    if metric == 'f1':
        dim =targets.shape[1]
        return np.array([f1(targets.reshape(-1,dim)[:,i], preds.reshape(-1,dim)[:,i]) for i in range(dim)])
    
    if metric == 'gmeans':
        return gmeans(targets, preds)
    
    if metric=='spe':
        return specificity(targets, preds)

    if metric == 'npv':
        return negPredValue(targets, preds)
    
    if metric == 'ccr':
        dim =targets.shape[1]
        return np.array([ccr(targets.reshape(-1,dim)[:,i], preds.reshape(-1,dim)[:,i]) for i in range(dim)])
    
    raise ValueError(f'Metric "{metric}" not supported.')


def get_metric_many(metric, targets, preds) -> tuple:
    if len(targets) == len(preds):
        metr = [get_metric(metric, target, pred) for target, pred in zip(targets, preds)]
    else:
        metr = [get_metric(metric, targets, pred) for pred in preds]
        
    u = np.mean(metr)
    std = np.std(metr)
    return u'{0:.3f} \u00B1 {1:.3f}'.format(u, std)
    # return str('%.3f'%u) + u" \u00B1 " + str('%.3f'%std)