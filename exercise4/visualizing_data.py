import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
X=np.load("Xtrain_Classification2.npy")
y=np.load("ytrain_Classification2.npy")
X_test=np.load("Xtest_Classification2.npy")

X=X.reshape(X.shape[0],28,28,3)

print(np.unique(y),"\n",np.round(100*(np.array(np.bincount(np.array(y,dtype=np.int64)),dtype=np.float32)/X.shape[0]),1))
choices=sorted(np.arange(0,X.shape[0],1))
num_of_samples=100

indexes=random.sample(choices,num_of_samples)
print(indexes)
fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(20, 14),
                        subplot_kw={'xticks': [], 'yticks': []})


for ax, ind in zip(axs.flat, indexes):
    ax.imshow(X[ind])
    ax.set_title(f"Classified as {int(y[ind])}")


plt.tight_layout()
plt.savefig(f"images/{num_of_samples}SamplesVisualization.png")
