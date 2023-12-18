from si.src.data.dataset import Dataset
from si.src.metrics.accuracy import accuracy

import numpy as np
import os
os.chdir("C:/Users/zemar/OneDrive/Documentos")


class CategoricalNB:
    '''
    this class computes a model that implements the Naive Bayes algorithm for categorical data. This class will be used only for binary features
    '''

    def __init__(self, smoothing: float = 1.0):
        '''
        Initialize the model

        Parameters:
        smoothing -> Smoothing parameter to avoid zero probabilities

        Attributes:
        class_prior ->Prior probabilities for each class
        feature_prob -> Conditional probabilities for each feature given the class
        '''
        self.smoothing = smoothing
        self.class_prior = list()
        self.feature_prob = list()

    def fit(self, dataset: Dataset) -> 'CategoricalNB':
        '''
        this function fits the model using the given dataset.

        Parameters:
        dataset -> The dataset to fit the model with
        
        Returns:
        self -> CategoricalNB
        '''
        n_samples, n_features = dataset.X.shape
        n_classes = len(dataset.get_classes())

        # initialize class_count (number of samples for each class), feature_count (sum of each feature for each class)
        # and class_prior (prior probability for each class)
        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))
        self.class_prior = np.zeros(n_classes)

        #smoothing
        for class_ in range(n_classes):
            class_count[class_] = np.sum(dataset.y == class_) + self.smoothing

        #feature_count
        for class_ in range(n_classes):
            for feature in range(n_features):
                feature_count[class_, feature] = np.sum(dataset.X[dataset.y == class_, feature], axis=0) + self.smoothing
        #class_prior
        self.class_prior = class_count / n_samples
        #feature_prob
        self.feature_prob = feature_count / class_count.reshape(-1, 1)

        return self       

        
    

    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        this function predicts the labels of the given dataset.

        Parameters: 
        dataset ->The dataset to predict the labels of

        Returns:
        y_pred -> The predicted labels
        '''
        n_classes = len(dataset.get_classes())
        class_prob = np.zeros(n_classes)
        y_pred = np.zeros(len(dataset.X))
        
        for i, sample in enumerate(dataset.X):
            for j in range(n_classes):
                class_prob[j] = np.prod(sample * self.feature_prob[j] + (1 - sample) * (1 - self.feature_prob[j])) * self.class_prior[j]
            y_pred[i] = np.argmax(class_prob)

        return y_pred


    def score(self, dataset: Dataset) -> float:
        '''
        this function Compute the score of the model on the given dataset.

        Parameters:
        dataset -> The dataset to compute the score on

        Returns:
        score -> The score of the model on the given dataset
        '''
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)




if __name__ == '__main__':
    from si.src.model_selection.split import train_test_split
    np.random.seed(42)
    dataset = Dataset.from_random(100, 10, 2)
    dataset_train, dataset_test = train_test_split(dataset, 0.2)

    model = CategoricalNB()
    model.fit(dataset_train)
    print('Model accuracy:', model.score(dataset_test))

    #comparison with sk
    from sklearn.naive_bayes import CategoricalNB as CategoricalNB_sk
    model_sk = CategoricalNB_sk()
    model_sk.fit(dataset_train.X, dataset_train.y)
    print('Model accuracy (sklearn):', model_sk.score(dataset_test.X, dataset_test.y))