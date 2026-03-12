import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc


def evaluate_svc():

    # ==============================
    # Load Data
    # ==============================

    xtrain = pd.read_csv(r'D:\Suchit\Deep-Learning\src\xtrain.csv', header=None)
    ytrain = pd.read_csv(r'D:\Suchit\Deep-Learning\src\ytrain.csv', header=None).values.ravel()

    xtest = pd.read_csv(r'D:\Suchit\Deep-Learning\src\xval.csv', header=None)
    ytest = pd.read_csv(r'D:\Suchit\Deep-Learning\src\yval.csv', header=None).values.ravel()


    # ==============================
    # Model
    # ==============================

    svc_model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearSVC(random_state=42))
    ])


    # ==============================
    # Train
    # ==============================

    svc_model.fit(xtrain, ytrain)


    # ==============================
    # Predictions
    # ==============================

    y_pred = svc_model.predict(xtest)


    # Decision scores for PR-AUC
    y_scores = svc_model.decision_function(xtest)


    # ==============================
    # Metrics
    # ==============================

    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    precision_curve, recall_curve, _ = precision_recall_curve(ytest, y_scores)
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