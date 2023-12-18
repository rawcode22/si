from typing import Callable, Union, Literal

import numpy as np

from si.src.data.dataset import Dataset
from si.src.metrics.rmse import rmse
from si.src.statistics.euclidean_distance import euclidean_distance

class KNNRegressor:
    '''
    this is a KNN Regressor class
    the k-nearest neighbors regressor is a supervised learning algorithm that predicts the value of a new sample based on the k-nearest samples in the training data.
    this class is similar to the KNNClassifier, but is more suitable for regression problems.
    it estimates the average value of the k most similar examples instead of the most common class.
    
    parameters:
    k -> The number of nearest neighbors to use
    distance -> The distance function to use
    
    attributes:
    dataset -> The training data
    '''

    def __init__(self, k: int = 1, weights: Literal['uniform', 'distance'] = 'uniform',  distance: Callable = euclidean_distance):
        '''
        this function initializes the KNN regressor

        parameters:
        k -> The number of nearest neighbors to use
        weights -> The weight function to use
        distance -> The distance function to use
        '''
        self.k = k
        self.distance = distance
        self.weights = weights
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        '''
        this function fits the model to the given dataset

        Parameters:
        dataset -> The dataset to fit the model to

        returns:
        self -> The fitted model
        '''
        self.dataset = dataset
        return self
    
    def _get_weights(self, distances: np.ndarray) -> np.ndarray:
        '''
        this function returns the weights of the k nearest neighbors

        parameters:
        distances -> The distances between the sample and the dataset

        returns: 
        weights -> The weights of the k nearest neighbors
        '''
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        distances[k_nearest_neighbors] = np.maximum(distances[k_nearest_neighbors], 0.000001)

        weights = 1 / distances[k_nearest_neighbors]
        return weights
    
    def _get_weighted_label(self, sample: np.ndarray) -> Union[int, str]:
        '''
        this function returns the weighted label of the most similar sample in the dataset

        parameters: 
        sample -> The sample to predict

        returns: 
        label -> The weighted label of the most similar sample in the dataset
        '''
        distances = self.distance(sample, self.dataset.X)
        weights = self._get_weights(distances)
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]
        label = np.sum(k_nearest_neighbors_labels * weights) / np.sum(weights)
        return label
    
    
    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        '''
        this function returns the label of the most similar sample in the dataset

        parameters:
        sample -> The sample to predict

        returns: 
        label -> The label of the most similar sample in the dataset
        '''
        if self.weights == 'distance':
            return self._get_weighted_label(sample)
        
        else:
            
            distances = self.distance(sample, self.dataset.X)
            k_nearest_neighbors = np.argsort(distances)[:self.k]
            k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]
            label = np.mean(k_nearest_neighbors_labels)
            return label
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        this function predicts the labels of the given dataset

        parameters:
        dataset -> The dataset to predict

        returns:
        y_pred -> The predicted labels
        '''
        y_pred = np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
        return y_pred
    
    def score(self, dataset: Dataset) -> float:
        '''
        this function returns the accuracy of the model on the given dataset

        parameters: 
        dataset -> The dataset to score

        returns: 
        accuracy ->  The RMSE of the model
        '''
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)