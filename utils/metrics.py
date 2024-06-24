import numpy as np

def f1_calc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_pred == True) & (y_true == True))
    FP = np.sum((y_pred == True) & (y_true == False))
    FN = np.sum((y_pred == False) & (y_true == True))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Function to calculate macro-average precision, recall, and F1 score
def precision_recall_f1(preds, labels, num_labels):
    preds = np.array(preds)
    labels = np.array(labels)

    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(num_labels):
        modified_true = (labels == i)
        modified_pred = (preds == i)
        precision, recall, f1 = f1_calc(modified_true, modified_pred)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    macro_precision = np.mean(precision_list)
    macro_recall = np.mean(recall_list)
    macro_f1 = np.mean(f1_list)

    return macro_precision, macro_recall, macro_f1
