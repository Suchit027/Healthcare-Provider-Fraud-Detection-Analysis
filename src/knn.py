import os

# Set thread limits BEFORE importing numpy/pandas
os.environ["OPENBLAS_NUM_THREADS"] = "24"
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "24"

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def evaluate_knn():

    # ==============================
    # Load Data
    # ==============================

    xtrain = pd.read_csv(r'D:\Suchit\Deep-Learning\src\xtrain.csv', header=None).values
    ytrain = pd.read_csv(r'D:\Suchit\Deep-Learning\src\ytrain.csv', header=None).values.ravel()

    xtest = pd.read_csv(r'D:\Suchit\Deep-Learning\src\xval.csv', header=None).values
    ytest = pd.read_csv(r'D:\Suchit\Deep-Learning\src\yval.csv', header=None).values.ravel()


    # ==============================
    # Model
    # ==============================

    knn_model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    ])


    # ==============================
    # Train
    # ==============================

    knn_model.fit(xtrain, ytrain)


    # ==============================
    # Predictions
    # ==============================

    y_pred = knn_model.predict(xtest)

    y_prob = knn_model.predict_proba(xtest)[:, 1]


    # ==============================
    # Metrics
    # ==============================

    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    precision_curve, recall_curve, _ = precision_recall_curve(ytest, y_prob)
    pr_auc = auc(recall_curve, precision_curve)


    # ==============================
    # Return dictionary
    # ==============================

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc
    }