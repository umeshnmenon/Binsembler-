from __future__ import division
import pandas as pd
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, f1_score, roc_auc_score, roc_curve


def get_optimal_threshold(y, y_preds):
    """
    Calculates the optimal threshold based on roc values
    :param y_preds:
    :return:
    """
    # AUCPR
    precision, recall, thresholds = precision_recall_curve(y, y_preds)
    auprc = auc(recall, precision)
    max_f1 = 0
    for r, p, t in zip(recall, precision, thresholds):
        if p + r == 0: continue
        if (2 * p * r) / (p + r) > max_f1:
            max_f1 = (2 * p * r) / (p + r)
            max_f1_threshold = t
    # logging.info('Optimal Threshold ' + str(max_f1_threshold) + ' for Maximized F1 @' + str(max_f1))
    return max_f1_threshold


def get_cm(y_pred, y_act, bins):
    """
    Calculates the confusion matrix for each bin
    :param y_pred:
    :param y_act:
    :param bins:
    :return:
    """
    cm = pd.DataFrame()
    cm['y_preds'] = y_pred
    cm['y_true'] = y_act
    cm['bins'] = bins
    cm['correctly_predicted'] = np.where(cm['y_true'] == cm['y_preds'], 1, 0)
    cm['TP'] = np.where((cm['y_true'] == 1) & (cm['y_preds'] == 1), 1, 0)
    cm['TN'] = np.where((cm['y_true'] == 0) & (cm['y_preds'] == 0), 1, 0)
    cm['FP'] = np.where((cm['y_true'] == 0) & (cm['y_preds'] == 1), 1, 0)
    cm['FN'] = np.where((cm['y_true'] == 1) & (cm['y_preds'] == 0), 1, 0)
    temp = cm.groupby(['bins']).agg({'TP': 'sum', 'TN': 'sum', 'FP': 'sum', 'FN': 'sum'}).reset_index()
    return temp[['bins', 'TP', 'TN', 'FP', 'FN']]


def get_accuracy(df):
    """
    Calculates the accuracy. Expects a DF object with columns TP, FP, TN, FN
    :param df:
    :return:
    """
    if (df['TP'] + df['TN'] + df['FP'] + df['FN']) == 0:
        return 0
    else:
        return (df['TP'] + df['TN']) / (df['TP'] + df['TN'] + df['FP'] + df['FN'])


def get_precision(df):
    """
    Calculates the precision. Expects a DF object with columns TP, FP, TN, FN
    :param df:
    :return:
    """
    if (df['TP'] + df['FP']) == 0:
        return 0
    else:
        return df['TP'] / (df['TP'] + df['FP'])


def get_recall(df):
    """
    Calculates the recall. Expects a DF object with columns TP, FP, TN, FN
    :param df:
    :return:
    """
    if (df['TP'] + df['FN']) == 0:
        return 0
    else:
        return df['TP'] / (df['TP'] + df['FN'])


def get_f1_score(df):
    """
    Calculates the f1 score. Expects a DF object with columns TP, FP, TN, FN
    :param df:
    :return:
    """
    if df['Precision'] + df['Recall'] == 0:
        return 0
    else:
        return 2 * ((df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall']))


def get_metrics(df):
    """
    Calculates all the evaluation metrics. Expects a DF object with columns TP, FP, TN, FN
    :param df:
    :return:
    """
    df['Accuracy'] = df['F1_Score'] = df['Precision'] = df['Recall'] = df["AUC"] = df["PR_AUC"] = None
    df['Accuracy'] = df.apply(get_accuracy, axis=1).values.tolist()
    df['Precision'] = df.apply(get_precision, axis=1).values.tolist()
    df['Recall'] = df.apply(get_recall, axis=1).values.tolist()
    df['F1_Score'] = df.apply(get_f1_score, axis=1).values.tolist()
    return df
