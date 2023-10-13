import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageChops

X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #0.8 - 0.2 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


X_train=X_train.reshape(X_train.shape[0],28,28,3) #3752

print(X_train.shape)
X_train_neg=X_train[y_train==0] #3218
X_train_pos=X_train[y_train==1] #531





#print(X_train_neg.shape,X_train_pos.shape)
#print(y_train.shape)

# X_train_pos_extra=[]
# for image in X_train_pos:
#     #Rotation to 90 and 180
#     for k in [1,2,3]:
#         X_train_pos_extra.append(np.rot90(image,k=k))
#     for axis in [0,1]:
#         X_train_pos_extra.append(np.flip(image,axis=axis))
# #print(np.array(X_train_pos_extra).shape)
# X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
# #print(X_train_pos.shape)
# X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
# #print(X_train.shape)
# y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
# #print(y_train.shape)




# 5 Data transformations visualization
# 1
image=X_train_pos[0]
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

for idx,_ in enumerate(axs.flat):
    print(idx)
    if idx<3:
        axis=0
    else:
        axis=1
    for p,s in enumerate([0,3,-3]):
        axs[axis,p].imshow(np.roll(image,shift=s,axis=axis))
        axs[axis,p].set_title(f"Shifted {s} positions in axis {axis}")
plt.tight_layout()
plt.show()
