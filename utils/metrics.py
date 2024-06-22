import numpy as np


def f1_calc(y_true, y_pred):
    TP = np.sum(np.multiply([i == True for i in y_pred], y_true))
    FP = np.sum(np.multiply([i == True for i in y_pred], [not(j) for j in y_true]))
    FN = np.sum(np.multiply([i == False for i in y_pred], y_true))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if precision != 0 and recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1

# Function to calculate macro-average precision, recall, and F1 score
def precision_recall_f1(preds, labels, num_labels):
    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(num_labels):
        modified_true = [i == j for j in labels]
        modified_pred = [i == j for j in preds]
        precision, recall, f1 = f1_calc(modified_true, modified_pred)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    macro_precision = np.mean(precision_list)
    macro_recall = np.mean(recall_list)
    macro_f1 = np.mean(f1_list)
    return macro_precision, macro_recall, macro_f1
