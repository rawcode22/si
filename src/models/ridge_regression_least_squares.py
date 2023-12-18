import numpy as np
import os
os.getcwd()
os.chdir("C:/Users/zemar/OneDrive/Documentos")
from si.src.data.dataset import Dataset
from si.src.metrics.mse import mse


class RidgeRegressionLeastSquares:
    '''
    this class implements a  Ridge Regression Least Squares model is a linear model that uses the Least Squares method to fit the data. 
    this model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm.
    
    parameters:
    l2_penalty -> The L2 regularization parameter
    scale -> Whether to scale the dataset or not
        
    Attributes:
    theta -> The model parameters, namely the coefficients of the linear model.  x0 * theta[0] + x1 * theta[1] + ...
    theta_zero -> The model parameter, namely the intercept of the linear model.  theta_zero * 1
    '''

    def __init__(self, l2_penalty: float = 1, scale: bool = True):
        '''
        parameters:
        l2_penalty -> The L2 regularization parameter
        scale -> Whether to scale the dataset or not               
        
        '''
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        '''
        this function fits the model using the given dataset.

        parameters:
        dataset -> The dataset to fit the model with

        returns:
        self -> The fitted model
        '''
        # scale the dataset
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X = np.c_[np.ones(X.shape[0]), X]
        penalty_matrix = np.eye(X.shape[1]) * self.l2_penalty

        penalty_matrix[0, 0] = 0

        theta_vector = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(dataset.y)
        self.theta_zero = theta_vector[0]
        self.theta = theta_vector[1:]

        return self
    
    def predict(self, dataset: Dataset) -> np.array:
        '''
        this function predicts the labels of the given dataset.

        parameters:
        dataset -> The dataset to predict the labels of

        returns:
        y_pred -> The predicted labels
        '''
       
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X = np.c_[np.ones(X.shape[0]), X]
        y_pred = X.dot(np.r_[self.theta_zero, self.theta])

        return y_pred
    
    def score(self, dataset: Dataset) -> float:
        '''
        this function computes the score of the model on the given dataset.

        parameters:
        dataset -> The dataset to compute the score on

        returns:
        score -> The score of the model on the given dataset
        '''
        return mse(dataset.y, self.predict(dataset))



if __name__ == '__main__':
   
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    
    model = RidgeRegressionLeastSquares()
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

  
    print(model.score(dataset_))

    
    from sklearn.linear_model import Ridge
    model = Ridge()
   
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_)
    print(model.intercept_) 
    print(mse(dataset_.y, model.predict(X)))
