import os

# Set thread limits BEFORE importing numpy/pandas
os.environ["OPENBLAS_NUM_THREADS"] = "24"
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "24"

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
xtrain = pd.read_csv(r'D:\Suchit\Deep-Learning\src\xtrain.csv', header=None).values
ytrain = pd.read_csv(r'D:\Suchit\Deep-Learning\src\ytrain.csv', header=None).values.ravel()

xtest = pd.read_csv(r'D:\Suchit\Deep-Learning\src\xval.csv', header=None).values
ytest = pd.read_csv(r'D:\Suchit\Deep-Learning\src\yval.csv', header=None).values.ravel()

# Model
lof = LocalOutlierFactor(
    n_neighbors=20,
    novelty=True,
    n_jobs=-1
)

# Train
lof.fit(xtrain)

# Predict
y_pred = lof.predict(xtest)

# Convert sklearn output (-1 anomaly → 1)
y_pred = (y_pred == -1).astype(int)

# Results
print("\nLocal Outlier Factor")
print("Accuracy:", accuracy_score(ytest, y_pred) * 100)
print("Precision:", precision_score(ytest, y_pred) * 100)
print("Recall:", recall_score(ytest, y_pred) * 100)
print("F1:", f1_score(ytest, y_pred) * 100)