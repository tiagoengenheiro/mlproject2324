import numpy as np
import random
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN, KMeansSMOTE,BorderlineSMOTE

X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy")

X=X.reshape(X.shape[0],28,28,3)
X=X[y==1]
image=X[17] #8,9
t={
    "Regular":image,
    "Rotation":np.rot90(image,k=1),
    "Shift":np.roll(image,shift=2,axis=1),
    "Flip":np.flip(image,axis=1)
}
for key in t:
    plt.imshow(t[key])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"report/{key}.png")




