from typing import Callable

import numpy as np

from si.src.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile:
    """
    Select features according to a specified percentile of the highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: int or float, default=10
        Percentile of features to select. Should be in the range (0, 100].

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, percentile: float = 10.0):
        """
        Select features according to a specified percentile of the highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: int or float, default=10
            Percentile of features to select. Should be in the range (0, 100].
        """
        if not 0 < percentile <= 100:
            raise ValueError("Percentile should be in the range (0, 100]")
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the top percentile of scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the selected percentile of scoring features.
        """
        num_features = dataset.X.shape[1]
        k = int(np.ceil(num_features * (self.percentile / 100.0)))
        idxs = np.argsort(self.F)[-k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectPercentile and transforms the dataset by selecting the top percentile of scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the selected percentile of scoring features.
        """
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':
    from si.src.data.dataset import Dataset
    
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    print(len(dataset.features))
    selector = SelectPercentile(percentile=20)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)