import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


X_train = np.load("Xtrain_Classification1.npy")
y_train = np.load("ytrain_Classification1.npy")
X_test = np.load("Xtest_Classification1.npy")

from sklearn import svm

svm_classifier = svm.SVC(kernel='rbf', C=1, gamma='scale',random_state=42)
print(X_train.shape,y_train.shape)
X_train,y_train=self_augmentation_rotate_flip(X_train,y_train)
print(X_train.shape,y_train.shape)
X_train = X_train.reshape(X_train.shape[0], -1)
svm_classifier.fit(X_train,y_train)

pred=svm_classifier.predict(X_test)

print(pred.shape)

np.save("ytest_Classification1.npy",pred)
y_pred=np.load("ytest_Classification1.npy")
print(y_pred.shape)
print(len(y_pred[y_pred==0]),len(y_pred[y_pred==1]))
