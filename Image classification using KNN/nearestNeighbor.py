import numpy as np

class NearestNeighbor:
    def __init__(self) -> None:
        pass
    
    def train(self, X, y):
        """X has entities with its attributes; row represents entities and columns represent the features. 
        y contains class of each entity."""
        self.xtr = X
        self.ytr = y

    def predict(self, X):
        """X has entities with its attributes; row represents entities and columns represent the features."""
        rows = X.shape[0]
        pred = np.zeros(rows)