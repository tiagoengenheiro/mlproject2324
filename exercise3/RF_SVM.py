import numpy as np
from sklearn.model_selection import train_test_split
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.utils import resample
# from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.model_selection import cross_val_score



def preprocessing(X_array, y_array, technique: str):
    if technique == 'oversampling':
        # sm = SMOTE(random_state=42)
        # X_array,y_array=sm.fit_resample(X_array,y_array)
        print("Enfim")

    elif technique == 'undersampling':
        maj_class = np.where(y == 0)[0]
        min_class = np.where(y == 1)[0]
        maj_undersampled = resample(maj_class, n_samples=len(min_class), random_state=42)
        X_array = X_array[np.concatenate([min_class, maj_undersampled])]
        y_array = y_array[np.concatenate([min_class, maj_undersampled])]

    elif technique == 'augmentation':
        X_array,y_array=self_augmentation_rotate_flip(X_array,y_array)
        #X_array,y_array=self_augmentation_shift(X_array,y_array)
        #X_array,y_array=self_augmentation(X_array,y_array)

    return X_array, y_array

if __name__ == "__main__":

    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = np.load("Xtrain_Classification1.npy")
    y = np.load("ytrain_Classification1.npy")

    X,y=self_augmentation_rotate_flip(X,y)
    X = X.reshape(X.shape[0], -1)

    print(X.shape)

    #Split dataset into train, val and test. (0.6, 0.2, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    X_ftrain, y_ftrain = X_train, y_train #Final training dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=30) 

    # RF Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100,criterion = 'entropy', random_state=1)
    rf_classifier.fit(X_ftrain, y_ftrain)
    y_pred = rf_classifier.predict(X_test)

    print("RF Test Results:")
    scores=cross_val_score(rf_classifier, X, y, cv=5, scoring=('balanced_accuracy'))
    print(scores)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Balanced Accuracy: {(100*balanced_acc):>0.1f}%, Recall:{(100*recall):>0.1f}%\n")

    # SVM Classifier
    svm = svm.SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(X_ftrain, y_ftrain)
    y_pred = svm.predict(X_test)

    print("SVM Test Results:")
    scores=cross_val_score(svm, X, y, cv=5, scoring=('balanced_accuracy'))
    print(scores)
    balanced_ac = balanced_accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Balanced Accuracy: {(100*balanced_acc):>0.1f}%, Recall:{(100*recall):>0.1f}%\n")





