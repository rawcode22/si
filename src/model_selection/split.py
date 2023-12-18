from typing import Tuple

import numpy as np

from si.src.data.dataset import Dataset
import os


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    this is a function which its purpose is to split the dataset into training and testing sets

    parameters:
    dataset -> The dataset to split
    test_size -> The proportion of the dataset to include in the test split
    random_state -> The seed of the random number generator

    returns:
    train -> The training dataset
    test -> The testing dataset
    """
  
    np.random.seed(random_state)
    n_samples = dataset.shape()[0]
    n_test = int(n_samples * test_size)
    permutations = np.random.permutation(n_samples)
    test_idxs = permutations[:n_test]
    train_idxs = permutations[n_test:]
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets preserving the proportion of samples per class
    
    Parameters:
    dataset ->  The dataset to split
    test_size -> The proportion of the dataset to include in the test split
    random_state -> The seed of the random number generator

    returns: 
    train -> The training dataset
    test -> The testing dataset
    """

    #set random state
    np.random.seed(random_state)
    
    #get unique class labels and their counts
    labels, counts = np.unique(dataset.y, return_counts=True)

    # initialize empty lists for the training and testing indices
    train_idxs = []
    test_idxs = []

    # loop through the unique labels
    for label in labels:
        # calculate the number of samples in the test set for the current label
        n_test = int(counts[np.where(labels == label)] * test_size)
        # shuffle and select the indices of the samples with the current label and add them to the testing list
        permutations = np.random.permutation(np.where(dataset.y == label)[0])
        test_idxs.extend(permutations[:n_test])
        # add the remaining indices to the training list
        train_idxs.extend(permutations[n_test:])

    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


