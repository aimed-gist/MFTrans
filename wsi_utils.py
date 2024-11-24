from monai.transforms import *
import os
import numpy as np
from model.MFNet import MFNet


def data_load(dataset_dir, end='tif'):
    file_list = []

    for subdir, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(f".{end}"):
                file_list.append(os.path.join(subdir, file))

    sorted_list = sorted(file_list, key=lambda x: os.path.basename(x))

    return sorted_list

def bring_model(archi,ds=True,num_tokens=10):

    model = MFNet(Global_branch="GlobalTokenTransformer",ds=ds,num_tokens=num_tokens)
    return model


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred, axis=(1, 2))
    denominator = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2))
    dice_score = np.divide(2. * intersection, denominator, out=np.ones_like(intersection, dtype=float), where=denominator!=0)
    return np.mean(dice_score)

def jaccard_index(y_true, y_pred):
    intersection = np.sum(y_true * y_pred, axis=(1, 2))
    union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2)) - intersection
    jaccard_score = np.divide(2. * intersection, union, out=np.ones_like(union, dtype=float), where=union!=0)
    return np.mean(jaccard_score)

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=(1, 2))
    tn = np.sum((y_true == 0) & (y_pred == 0), axis=(1, 2))
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=(1, 2))
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=(1, 2))
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.ones_like(tp, dtype=float), where=(tp + fn) != 0)
    f1_score = np.divide(2 * precision * recall, precision + recall, out=np.ones_like(precision, dtype=float), where=(precision + recall) != 0)
    accuracy = np.divide(tp + tn, tp + tn + fp + fn, out=np.ones_like(tp, dtype=float), where=(tp + tn + fp + fn) != 0)
    jaccard=jaccard_index(y_true, y_pred)
    dice = dice_coefficient(y_true, y_pred)
    return jaccard, dice,  np.mean(f1_score), np.mean(accuracy)