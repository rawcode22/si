import numpy as np
from si.src.data.dataset import Dataset
from typing import Union

class PCA:
    '''
    This class computes the Principal Component Analysis (PCA) on a dataset.
    Groups the data into a n number of components.

    Parameters:
    
    n_components: int
      
    Attributes;
    
    components: np.ndarray
        
    mean: np.ndarray
      
    explained_variance: np.ndarray
        
    '''
    def __init__(self, n_components: Union[int, float]):
        
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_var = None
        self.explained_var_ratio = None


    def _get_centered_data(self, dataset: Dataset) -> np.ndarray:
        '''
        This function centers the data.

        Parameters:
        
        dataset: Dataset
           
        Returns:
        
        np.ndarray: Centered data
            
        '''
        self.mean = np.mean(dataset.X, axis=0) # mean of each feature
        return dataset.X - self.mean
    
    def _get_components(self, dataset:Dataset) -> np.ndarray:
        '''
        This function fetches the components from the given centered data.
        
        Returns:
        
        np.ndarray: Components
        '''
        #singular value decomposition
        self.U, self.S, self.V = np.linalg.svd(self._get_centered_data(dataset), full_matrices=False)

        #extracting the components (first n_components columns of V)
        self.components = self.V[:self.n_components]

        return self.components
    
    def _get_explained_variance(self, dataset:Dataset) -> np.ndarray:
        '''
        Calculates the explained variance of the centered data.

        Returns:
      
        np.ndarray: Explained variance.
        '''

        #extracting the explained variance
        ev = (self.S ** 2) / (self._get_centered_data(dataset).shape[0] - 1)
        self.explained_var = ev[:self.n_components]
        self.explained_var_ratio = self.explained_var / np.sum(self.explained_var)

        return self.explained_var
    
    def fit(self, dataset: Dataset) -> 'PCA':
        '''
        It fits the model.

        Parameters_
       
        dataset: Dataset
        
        Returns:
        
        Results of PCA
            
        '''
        
        if 0 < self.n_components < 1.0:
            # calculate the explained variance ratio 
            full_pca = PCA(n_components=dataset.X.shape[1])
            full_pca.fit(dataset)
            ratio_cumsum = np.cumsum(full_pca.explained_var_ratio)
            
            # shortens up the number of components to the number that explains the given variance
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
        It fits and transforms the dataset.

        Parameters:
  
        dataset: Dataset

        Returns:
        
        Trasformed dataset
        '''
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':
    from si.src.io.csv_file import read_csv
    iris = read_csv('datasets/iris/iris.csv',sep=',',features=True, label=True)

    pca = PCA(n_components=0.95)
    pca.fit_transform(iris)
    print(pca.explained_var_ratio)

    from sklearn.decomposition import PCA as PCA_sklearn
    pca_sklearn = PCA_sklearn(n_components=0.95)
    pca_sklearn.fit_transform(iris.X)
    print(pca_sklearn.explained_var_ratio_)
