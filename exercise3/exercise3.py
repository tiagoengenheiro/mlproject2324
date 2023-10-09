
import numpy as np
from utils import get_best_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy").reshape(X.shape[0],1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #0.8 - 0.2 

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
#0.6 Train e 0.2 Val e 0.2 Test

n_examples_train,n_features=X_train.shape
n_examples_test,_=X_test.shape
n_examples_val,_=X_val.shape
print(n_examples_train,n_examples_val,n_examples_test)
model = Sequential()
model.add(Dense(256, input_shape=(n_features,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))
print(model.output_shape)
sgd = SGD(1e-4)
model.compile(loss="binary_crossentropy", optimizer=sgd,metrics=["accuracy"])
H = model.fit(X_train, y_train, validation_data=(X_val, y_val),
	epochs=100, batch_size=128)
print(H.history)

predictions = model.predict(X_test, batch_size=128)
print(predictions)