#https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
import numpy as np
from utils import get_best_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN, KMeansSMOTE,BorderlineSMOTE
import tensorflow
X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy")


from sklearn.utils import class_weight

# Compute class weights



#sm = KMeansSMOTE(random_state=42,kmeans_estimator=) 
# BorderlineSmote: 0.8079379562043796
# ADASYN - Balanced Accuracy: 0.7925613527627089
#  SMOTE -  #Balanced Accuracy: 0.7802522705744692

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #0.8 - 0.2 

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train),y=y_train)

# Convert class weights to a dictionary for use in training
class_weight_dict = dict(enumerate(class_weights))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
sm = SMOTE(random_state=42)
X_train,y_train=sm.fit_resample(X_train,y_train)
#0.6 Train e 0.2 Val e 0.2 Test
print("Split: Y","N_examples:",len(y),"Class 0:",len(y[y==0]),"Class 1:",len(y[y==1]),"Ratio:",len(y[y==0])/len(y[y==1]))
print("Split: Y_train",len(y_train),"Class 0:",len(y_train[y_train==0]),"Class 1:",len(y_train[y_train==1]),"Ratio:",len(y_train[y_train==0])/len(y_train[y_train==1]))
print("Split: Y_test",len(y_test),"Class 0:",len(y_test[y_test==0]),"Class 1:",len(y_test[y_test==1]),"Ratio:",len(y_test[y_test==0])/len(y_test[y_test==1]))
print("Split: Y_val",len(y_val),"Class 0:",len(y_val[y_val==0]),"Class 1:",len(y_val[y_val==1]),"Ratio:",len(y_val[y_val==0])/len(y_val[y_val==1]))

_,n_features=X_train.shape
model = Sequential()
model.add(Dense(256, input_shape=(n_features,), activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))
print(model.output_shape)
sgd = SGD(1e-4)
model.compile(loss="binary_crossentropy", optimizer=sgd,metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val),
	epochs=100, batch_size=128)

y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Calculate balanced accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print("Balanced Accuracy:", balanced_acc)
# predictions = model.predict(X_test, batch_size=128)
# print(predictions)