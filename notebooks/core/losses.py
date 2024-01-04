# Author: Ankit Kariryaa, University of Bremen



import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

def tversky(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits 
    :param ground_truth: the segmentation ground_truth  
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :return: the loss
    """  
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    tp = K.sum(y_true_f*y_pred_f)
    fp = alpha * K.sum(y_true_f * (1 - y_pred_f))
    fn = beta * K.sum((1 - y_true_f) * y_pred_f)
    numerator = tp + + K.epsilon()
    denominator = tp + fp + fn + K.epsilon()
    score = numerator / denominator
    return 1.0 - score

def focalTversky(y_true, y_pred,gamma=0.75):
    pt_1=tversky(y_true, y_pred)
    return K.pow((pt_1),gamma)

def dice_coef(y_true, y_pred):
    """compute dice coef"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_loss(y_true, y_pred):
    """compute dice loss"""
#     y_t = y_true[...,0]
#     y_t = y_t[...,np.newaxis]
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred, alpha=0.2, beta=0.8):
    return binary_crossentropy(y_true, y_pred)*alpha+dice_loss(y_true, y_pred)*beta

def true_positives(y_true, y_pred):
    """compute true positive"""
#     y_t = y_true[...,0]
#     y_t = y_t[...,np.newaxis]
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.round(y_t * y_pred)

def false_positives(y_true, y_pred):
    """compute false positive"""
#     y_t = y_true[...,0]
#     y_t = y_t[...,np.newaxis]
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.round((1 - y_t) * y_pred)

def true_negatives(y_true, y_pred):
    """compute true negative"""
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.round((1 - y_t) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    """compute false negative"""
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.round((y_t) * (1 - y_pred))

def accuracy(y_true, y_pred):#Calculates how often predictions equal labels.
    """compute accuracy"""
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.equal(K.round(y_t), K.round(y_pred))#Accuracy =  (TP + TN) / (TP + TN + FP+ FN) 

def IoU(y_t, y_pred):#the Intersection-Over-Union metric.
    # IoU = TP / (TP + FP + FN)
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp)/(K.sum(tp)+K.sum(fp)+K.sum(fn)+K.epsilon())
    

def recall(y_t, y_pred):#recall = TP / (TP + FN)
    """compute sensitivity (recall)"""
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn)+ K.epsilon())

def precision(y_t, y_pred):
    """compute precision"""
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fp)+ K.epsilon())

def F1_score(y_t, y_pred):
    re = recall(y_t, y_pred)
    pr = precision(y_t, y_pred)
    return 2*pr*re/(re+pr+K.epsilon())