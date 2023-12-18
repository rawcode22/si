import numpy as np
from si.src.data.dataset import Dataset
from si.src.metrics.accuracy import accuracy

class StackingClassifier:
    '''
    this class implements a stacking classifier that ensembles multiple models to formulate a prediction.
    '''

    def __init__(self, models: list, final_model: object) -> None:
        '''
        this function creates a new instance of the StackingClassifier class.

        Parameters:
        models -> he list of models to use.
        final_model -> The final model to use.
        '''
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        '''
        this function fits the model to the given data.

        parameters:
        dataset -> The dataset to fit the model to.

        returns:
        StackingClassifier -> The fitted model.
        '''
        for model in self.models:
            model.fit(dataset)

        predictions = list()
        for model in self.models:
            predictions.append(model.predict(dataset))
        
        #transpose the predictions (we do this because we want the models to be the columns and each prediction to be a row)
        predictions = np.array(predictions).T

        #train the final model with the predictions made by the other models
        self.final_model.fit(Dataset(dataset.X, predictions))

        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        this function predicts the labels for the given data.

        Parameters: 
        dataset -> The dataset to predict the labels for.

        Returns:
        np.ndarray -> The predicted labels.
        '''
        predictions = list()
        for model in self.models:
            predictions.append(model.predict(dataset))
        
        predictions = np.array(predictions)
        return self.final_model.predict(Dataset(dataset.X, predictions))
    
    def score(self, dataset: Dataset) -> float:
        '''
        this function calculates the accuracy of the model on the given data.

        parameters:
        dataset -> The dataset to calculate the accuracy for.

        returns:
        float -> The accuracy of the model.
        '''
        return accuracy(dataset.y, self.predict(dataset))

