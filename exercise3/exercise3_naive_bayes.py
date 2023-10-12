import numpy as np
from utils import get_best_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,recall_score
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN, KMeansSMOTE,BorderlineSMOTE
import tensorflow
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
print("Test","Recall:",recall_score(y_test,y_pred),"Balanced Acc:",balanced_accuracy_score(y_test,y_pred))

#Validation Recall: 0.8095238095238095 Balanced Acc: 0.7070217917675545


#Test Recall: 0.7687861271676301 Balanced Acc: 0.6803114309307539
#FFN com early stopping Recall e ADASYN: 0.838150289017341  Balanced Accuracy: 0.7029341426533829 
#CNN:  73.5%, Recall:93.1%, Avg loss: 0.667851, sem smote lr=e-3 e batch size=8
#CNN  com ADASYN Balanced Accuracy: 69.2%, Recall:79.2%, Avg loss: 0.688430 
