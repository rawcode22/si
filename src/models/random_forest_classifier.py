import numpy as np
from typing import List, Tuple
from si.src.models.decision_tree_classifier import DecisionTreeClassifier
from si.src.data.dataset import Dataset
from si.src.metrics.accuracy import accuracy

class RandomForestClassifier:
    '''
    this class implements a Random Forest Classifier model.
    '''

    def __init__(self, n_estimators: int, max_features: int = None, min_sample_split: int = 2, max_depth: int = 10, mode: str = 'gini', seed: int = 42) -> None:
        '''
        this class creates a new instance of the RandomForestClassifier class.

        Parameters:
        n_estimators -> The number of trees in the forest.
        max_features -> The number of features to consider when looking for the best split.
        min_sample_split -> The minimum number of samples required to split an internal node.
        max_depth -> The maximum depth of the tree.
        mode -> The impurity measure to use.
        seed -> The seed to use for the random number generator.
        '''
        
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        '''
        this function fits the model to the given data.

        Parameters:
        X ->  The input data.
        y -> The target values.

        returns:
        RandomForestClassifier -> The fitted model.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = dataset.shape()

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # loop through the number of estimators (trees)
        for i in range(self.n_estimators):
            bootstrap_samples_idx = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_features_idx = np.random.choice(n_features, self.max_features, replace=False)
            bootstrap_dataset = Dataset(dataset.X[bootstrap_samples_idx, :][:, bootstrap_features_idx], dataset.y[bootstrap_samples_idx])
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_sample_split=self.min_sample_split, mode=self.mode)
            tree.fit(bootstrap_dataset)
            self.trees.append((bootstrap_features_idx, tree))
        return self
    
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        this function predicts the target values for the given data.

        parameters: 
        dataset ->  The input data.

        returns:       
        np.ndarray -> The predicted target values.
        '''
        #array to store the predictions of each tree
        predictions = [None] * self.n_estimators

        # loop through the trees and get their predictions
        for i, (features_idx, tree) in enumerate(self.trees):
            predictions[i] = tree.predict(Dataset(dataset.X[:, features_idx], dataset.y))
    
        most_frequent = []
        for z in zip(*predictions):
            most_frequent.append(max(set(z), key=z.count))

        return np.array(most_frequent)

    def score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters:
        dataset -> The dataset to calculate the accuracy on.

        returns:
        float -> The accuracy of the model on the dataset.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


