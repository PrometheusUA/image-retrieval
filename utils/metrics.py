import numpy as np
from typing import Optional


def mean_average_precision_at_k(y_true:np.ndarray, y_pred:np.ndarray, k:int=10, cutoff: Optional[float]=None) -> float:
    """
    Calculate mean average precision at k
    Args:
        y_true: an integer numpy array of shape (n_samples,)
            A list of true labels.
            Example: array([1, 1, 3])
        y_pred: an integer numpy 2d array of shape (n_samples, n_classes)
            Example: array([[0.1, 0.4, 0.8], [0.001, 0.001, 0.002], [0, 0.1, 0.3]])
        k: int
            the value of k
        cutoff: float
            the value of cutoff to find distractors
    Returns:
        float - value of MAP@k metric
    """
    y_pred_topk = np.argsort(y_pred, axis=1)[:, -k:][:, ::-1]
    if cutoff is not None:
        cutoff_location = np.sum(y_pred > cutoff, axis=1)
        # set y_pred_topk[cutoff_location:] to -1 if cutoff_location < k
        # classes following the cutoff should be shifted to the right
        for i in range(y_pred_topk.shape[0]):
            if cutoff_location[i] < k:
                for j in range(1, k - cutoff_location[i]):
                    y_pred_topk[i, k - j] = y_pred_topk[i, k - j - 1]
                y_pred_topk[i, cutoff_location[i]] = -1
    
    aps = []
    for i in range(y_pred_topk.shape[0]):
        ap = 0
        for j in range(y_pred_topk.shape[1]):
            if y_pred_topk[i, j] == y_true[i]:
                ap = 1 / (j + 1)
        aps.append(ap)
    return np.mean(aps)