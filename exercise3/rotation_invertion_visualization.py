import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #0.8 - 0.2 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


X_train=X_train.reshape(X_train.shape[0],28,28,3) #3752

X_train_neg=X_train[y_train==0] #3218
X_train_pos=X_train[y_train==1] #534
#print(X_train_neg.shape,X_train_pos.shape)
#print(y_train.shape)

X_train_pos_extra=[]
for image in X_train_pos:
    #Rotation to 90 and 180
    for k in [1,2,3]:
        X_train_pos_extra.append(np.rot90(image,k=k))
    for axis in [0,1]:
        X_train_pos_extra.append(np.flip(image,axis=axis))
#print(np.array(X_train_pos_extra).shape)
X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
#print(X_train_pos.shape)
X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
#print(X_train.shape)
y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
#print(y_train.shape)




#5 Data transformations visualization
#1
# image=X_train_pos[0]
# fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(10, 6),
#                         subplot_kw={'xticks': [], 'yticks': []})

# for idx,ax in enumerate(axs.flat):
#     if idx<4:
#         ax.imshow(np.rot90(image,k=idx))
#         ax.set_title(f"Rotated {idx*90}º")
#     else:
#         ax.imshow(np.flip(image,axis=idx-4))
#         ax.set_title(f"Flipped in axis {idx-4}")
    
# plt.tight_layout()
# plt.show()
