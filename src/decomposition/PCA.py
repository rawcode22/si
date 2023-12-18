import numpy as np
from si.src.data.dataset import Dataset
from typing import Union

class PCA:
    '''
    this class computes the Principal Component Analysis (PCA) on a dataset.
    it groups the data into a n number of components.

    Parameters:

    n_components -> int
      
    Attributes:

    components -> np.ndarray
    mean -> np.ndarray 
    explained_variance -> np.ndarray
        
    '''
    def __init__(self, n_components: Union[int, float]):
        
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_var = None
        self.explained_var_ratio = None


    def _get_centered_data(self, dataset: Dataset) -> np.ndarray:
        '''
        this function centers the data.

        parameters:
        dataset -> Dataset
           
        returns:
        np.ndarray -> Centered data
            
        '''
        self.mean = np.mean(dataset.X, axis=0) # mean of each feature
        return dataset.X - self.mean
    
    def _get_components(self, dataset:Dataset) -> np.ndarray:
        '''
        this function fetches the components from the given centered data.
        
        returns:
        
        np.ndarray -> Components
        '''
        self.U, self.S, self.V = np.linalg.svd(self._get_centered_data(dataset), full_matrices=False)
        self.components = self.V[:self.n_components]

        return self.components
    
    def get_explained_variance(self, dataset:Dataset) -> np.ndarray:
        '''
        this function calculates the explained variance of the centered data.

        returns:
        np.ndarray -> Explained variance.
        '''
        ev = (self.S ** 2) / (self._get_centered_data(dataset).shape[0] - 1)
        self.explained_var = ev[:self.n_components]
        self.explained_var_ratio = self.explained_var / np.sum(self.explained_var)

        return self.explained_var
    
    def fit(self, dataset: Dataset) -> 'PCA':
        '''
        this function fits the model.

        parameters:
        dataset -> Dataset
        
        returns:
        Results of PCA
            
        '''
        
        if 0 < self.n_components < 1.0:
            full_pca = PCA(n_components=dataset.X.shape[1])
            full_pca.fit(dataset)
            ratio_cumsum = np.cumsum(full_pca.explained_var_ratio)
            
            self.n_components = np.searchsorted(ratio_cumsum, self.n_components) + 1
        
        self._get_components(dataset)
        self._get_explained_variance(dataset)
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        '''
        It transforms the dataset.

        Parameters:
        
        dataset: Dataset
        
        Returns:
     
        Transformed dataset
        '''
        
        X_reduced = np.dot(self._get_centered_data(dataset), self.V.T)
        return Dataset(X_reduced, dataset.y, dataset.features, dataset.label)
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        '''
        this function fits and transforms the dataset.

        parameters:
        dataset -> Dataset

        returns:
        Trasformed dataset
        '''
        self.fit(dataset)
        return self.transform(dataset)
    
