import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score
from utils import *


X=np.load("Xtrain_Classification2.npy")
y=np.load("ytrain_Classification2.npy")
#X_test=np.load("Xtest_Classification2.npy")


random_state=42
#No augmentation
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=random_state)

rf_classifier = RandomForestClassifier(n_estimators=100,criterion = 'entropy', random_state=random_state)
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale',random_state=random_state)
clfs={
    "RF":[],
    "SVM":[],
}
for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_train,y_train=X[train_index],y[train_index]
    X_train,y_train=self_augmentation_2(X_train,y_train)
    X_test,y_test=X[test_index],y[test_index]
    for i,clf in enumerate([rf_classifier,svm_classifier]):
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)
        b_acc=balanced_accuracy_score(y_test,pred)
        keys=list(clfs.keys())
        clfs[keys[i]].append(b_acc)
        # print([keys[i]],b_acc)

for key in clfs:
    print(f"{key} mean: {np.mean(clfs[key])}  std: {np.std(clfs[key])}")

