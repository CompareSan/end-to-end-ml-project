from typing import Tuple

import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_transform(
    tfidf_vect: TfidfVectorizer,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix, scipy.sparse.spmatrix]:
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    X_valid_tfidf = tfidf_vect.transform(X_valid)
    X_test_tfidf = tfidf_vect.transform(X_test)
    return X_train_tfidf, X_valid_tfidf, X_test_tfidf
