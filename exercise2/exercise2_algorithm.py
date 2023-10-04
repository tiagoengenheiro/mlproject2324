import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_best_model,fit_linear_models
from sklearn.preprocessing import RobustScaler,StandardScaler
X_train=np.load("X_train_regression2.npy") #shape -> (100, 4)
y_train=np.load("y_train_regression2.npy") #shape -> (100, 1)


n_examples,n_features=X_train.shape
seed=np.random.randint(0,10**3)
np.random.seed(seed)
np.random.shuffle(X_train)
np.random.seed(seed)
np.random.shuffle(y_train)
#Initialization (include randomness after)
X_train_1=X_train[:n_examples//2]
y_train_1=y_train[:n_examples//2]
X_train_2=X_train[n_examples//2:n_examples]
y_train_2=y_train[n_examples//2:n_examples]
counter=1
for i in range(10):
    X_train_1,y_train_1,X_train_2,y_train_2=fit_linear_models(X_train,y_train,X_train_1,y_train_1,X_train_2,y_train_2)
    print(counter,"1st Model",get_best_model(X_train_1,y_train_1)[1],"2nd Model",get_best_model(X_train_2,y_train_2)[1])
    counter+=1
model1=get_best_model(X_train_1,y_train_1)[0]
model2=get_best_model(X_train_2,y_train_2)[0]

#results=np.hstack((RS.inverse_transform(model1.predict(X_train_init)),RS.inverse_transform(model2.predict(X_train_init))))
results=np.hstack((model1.predict(X_train).reshape(100,1),model2.predict(X_train).reshape(100,1)))
results = np.square(results-y_train)
results = np.min(results,axis=1)
results = np.sum(results,axis=0)
print(results)




