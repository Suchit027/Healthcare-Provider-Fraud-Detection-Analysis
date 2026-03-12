from sklearn.covariance import EllipticEnvelope
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Load data
xtrain = pd.read_csv(r'D:\Suchit\Deep-Learning\src\xtrain.csv', header=None)
ytrain = pd.read_csv(r'D:\Suchit\Deep-Learning\src\ytrain.csv', header=None).values.ravel()

xtest = pd.read_csv(r'D:\Suchit\Deep-Learning\src\xval.csv', header=None)
ytest = pd.read_csv(r'D:\Suchit\Deep-Learning\src\yval.csv', header=None).values.ravel()

# Model
ee = EllipticEnvelope(contamination=0.1)

# Train
ee.fit(xtrain)

# Predict
y_pred = ee.predict(xtest)

# Convert sklearn output (-1 anomaly → 1)
y_pred = (y_pred == -1).astype(int)

print("\nElliptic Envelope")
print("Accuracy:", accuracy_score(ytest, y_pred) * 100)
print("Precision:", precision_score(ytest, y_pred) * 100)
print("Recall:", recall_score(ytest, y_pred) * 100)
print("F1:", f1_score(ytest, y_pred) * 100)

# Get anomaly scores
y_scores = -ee.decision_function(xtest)   # invert so higher = more anomalous

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(ytest, y_scores)

# PR-AUC
pr_auc = auc(recall, precision)
print("PR-AUC:", pr_auc)

# Plot PR curve
plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Elliptic Envelope)")
plt.legend()
plt.grid()
plt.show()