
import numpy as np
from utils import get_best_model
X_train=np.load("Xtrain_Classification1.npy")
y_train=np.load("ytrain_Classification1.npy")

print(X_train.shape)
print(y_train.shape) 
print(sum(y_train)) #896 are melanoma and and the rest (5000) is nevu. nevu>>melanoma

get_best_model(X_train,y_train)