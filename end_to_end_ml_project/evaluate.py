import numpy as np
from sklearn.metrics import accuracy_score


def get_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = model.predict(X)
    acc_score = accuracy_score(y, y_pred)

    return acc_score
