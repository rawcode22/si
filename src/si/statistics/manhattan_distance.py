import numpy as np

def manhattan_distance(x: np.ndarray, y:np.ndarray) -> np.ndarray:
    '''
    It calculates the manhattan distance of a point x to a group of y points.
    
        xtoy1 = |x1 - y11| + |x2 - y12| + ... + |xn - y1n|
        xtoy2 = |x1 - y21| + |x2 - y22| + ... + |xn - y2n|
        
    Parameters:
    
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns:
    
    np.ndarray
        Manhattan distance for each point in y.
    '''
    return np.abs((x-y).sum(axis=1))
