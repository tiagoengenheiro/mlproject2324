import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    X = np.load("Xtrain_Classification1.npy")
    y = np.load("ytrain_Classification1.npy")

    random_state=42

    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=random_state)

    rf_classifier = RandomForestClassifier(n_estimators=100,criterion = 'entropy', random_state=random_state)
    svm_classifier = svm.SVC(kernel='rbf', C=10, gamma='scale',random_state=random_state)

    clfs={
        "RF":[],
        "SVM":[],
    }
    
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train,y_train=X[train_index],y[train_index]
        print(X_train.shape,y_train[y_train==1].shape,y_train[y_train==0].shape)
        X_train,y_train=self_augmentation_rotate_flip(X_train,y_train)
        print(X_train.shape,y_train[y_train==1].shape,y_train[y_train==0].shape)
        X_train = X_train.reshape(X_train.shape[0], -1)
        print(X_train.shape)
        X_test,y_test=X[test_index],y[test_index]
        for i,clf in enumerate([rf_classifier,svm_classifier]):
            clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            b_acc=balanced_accuracy_score(y_test,pred)
            if i==0:
                clfs["RF"].append(b_acc)
            else:
                clfs["SVM"].append(b_acc)
        print(clfs)

print(clfs)
for key in clfs:
    print(f"{key} mean: {np.mean(clfs[key])}  std: {np.std(clfs[key])}")
