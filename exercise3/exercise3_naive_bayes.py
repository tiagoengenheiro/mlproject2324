import numpy as np
from utils import get_best_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,recall_score
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN, KMeansSMOTE,BorderlineSMOTE
from sklearn.naive_bayes import MultinomialNB,ComplementNB
X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy").reshape(X.shape[0],1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #0.8 - 0.2 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


clf = MultinomialNB(force_alpha=True)
print(X_train.shape,y_train.shape)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_val)
print("Validation","Recall:",recall_score(y_val,y_pred),"Balanced Acc:",balanced_accuracy_score(y_val,y_pred))

y_pred=clf.predict(X_test)
test_recall=recall_score(y_test,y_pred)
test_b_acc=balanced_accuracy_score(y_test,y_pred)
print("Test","Recall:",test_recall,"Specificity:",2*test_b_acc-test_recall, "Balanced Acc:",test_b_acc)

