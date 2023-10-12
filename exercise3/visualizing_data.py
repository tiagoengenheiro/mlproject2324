import numpy as np
import random
import matplotlib.pyplot as plt
X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy")
X=X.reshape(X.shape[0],28,28,3)
choices=sorted(np.arange(0,X.shape[0],1))
print(y[0])
num_of_samples=15
indexes=random.sample(choices,num_of_samples)

fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})


for ax, ind in zip(axs.flat, indexes):
    ax.imshow(X[ind])
    ax.set_title(f"Classified as {int(y[ind])}")


plt.tight_layout()
plt.savefig(f"images/{num_of_samples}SamplesVisualization.png")