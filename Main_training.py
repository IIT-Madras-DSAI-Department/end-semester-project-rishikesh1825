import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,precision_score, recall_score
import time
from sklearn.metrics import f1_score
from Algorithms_ML_lab import XGBoostClassifierMulticlass, XGBoostClassifierMulticlass_RandomForest, SoftmaxRegression, XGBoostClassifierMulticlass_OvR_RF


def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'label'

    Xtrain = np.array(dftrain[featurecols])/255
    ytrain = np.array(dftrain[targetcol])
    
    Xval = np.array(dfval[featurecols])/255
    yval = np.array(dfval[targetcol])

    return (Xtrain, ytrain, Xval, yval)

Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')

model = XGBoostClassifierMulticlass_OvR_RF(n_estimators=40, learning_rate=0.5, max_depth=5, n_bins=10, lam=0.5, feature_subsample_size=None)

start = time.time()

model.fit(Xtrain, ytrain)
y_pred = model.predict(Xval)

end = time.time()

print(f"{f1_score(yval, y_pred, average='micro'):.4f}")
