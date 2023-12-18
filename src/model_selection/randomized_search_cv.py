import numpy as np
from si.src.data.dataset import Dataset
from si.src.metrics.accuracy import accuracy
from si.src.model_selection.cross_validation import k_fold_cross_validation

def randomized_search_cv(model, dataset: Dataset, hyperparameter_grid: dict, scoring: callable = accuracy, cv: int = 5, n_ite: int = 10) -> dict:
    '''
    this class implements a parameter optimization strategy with cross validation using a random number of combinations from the hyperparameter grid.

    parameters: 
    model -> The model to cross validate.
    dataset -> The dataset to cross validate on.
    hyperparameter_grid -> The hyperparameter grid to use.
    scoring -> The scoring function to use. If None, the model's score method will be used.
    cv -> The number of cross-validation folds.
    n_ite ->  The number of iterations to perform.

    returns:
    best_params -> Dictionary with the results of the cross validation. Includes the scores, best parameters and the best model.
    '''
    if not isinstance(hyperparameter_grid, dict):
        raise TypeError('must be a dictionary.')
    
    for parameter in hyperparameter_grid.keys():
        if not hasattr(model, parameter):
            raise AttributeError(f"{model} does not have parameter {parameter}.")
    
    randomized_search_output = {'hyperparameters': [], 
                                'scores': [], 
                                'best_hyperparameters': None, 
                                'best_scores': 0}
        
    #n_iter hyperparameter combinations
    for i in range(n_ite):
        random_params = {}
        #random combination of hyperparameters
        for key in hyperparameter_grid.keys():
            random_params[key] = np.random.choice(hyperparameter_grid[key])

        #hyperparameters based on the random combination selected above
        for key in random_params.keys():
            setattr(model, key, random_params[key])

        #perform cross validation
        model_cv_scores = k_fold_cross_validation(model, dataset, scoring, cv)

        randomized_search_output['hyperparameters'].append(random_params)
        randomized_search_output['scores'].append(model_cv_scores)
        avg_score = np.mean(model_cv_scores)

        # check if the current model is the best one
        if avg_score > randomized_search_output['best_scores']:
            randomized_search_output['best_scores'] = avg_score
            randomized_search_output['best_hyperparameters'] = random_params
        
    return randomized_search_output


