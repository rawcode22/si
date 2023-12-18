from typing import Callable

import numpy as np

from si.src.data.dataset import Dataset
from si.src.statistics.f_classification import f_classification


class SelectPercentile:
    """
    this class selects features according to the specified percentile.
    the ranking is calculated by a score function, in this case f_classification.

    parameters:
    score_func -> Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile -> Number of top features to select.

    attributes:
    F -> F scores of features.
    p -> p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, percentile: int = 20):
        """
        this function selects features according to the percentile.

        parameters:
        score_func -> Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile -> Percentile of top features to select.
        """
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        parameters:
        dataset -> A labeled dataset

        returns:
        self -> Returns self.
        """
        
        self.F, self.p = self.score_func(dataset)
        self.F = np.nan_to_num(self.F)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        this function transforms the dataset by selecting the features in a specified percentile.

        parameters:
        dataset -> A labeled dataset

        returns:
        dataset -> A labeled dataset with the highest scoring features within a specified percentile.
        """

        threshold = np.percentile(self.F, 100 - self.percentile)
        idxs = np.where(self.F > threshold)[0]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectPercentile and transforms the dataset by selecting the highest scoring features within a specified percentile.

        parameters:
        dataset -> A labeled dataset

        returns:
        dataset -> A labeled dataset with the highest scoring features within a specified percentile.
        """
        self.fit(dataset)
        return self.transform(dataset)
    

