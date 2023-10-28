import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *

X=np.load("Xtrain_Classification2.npy")
y=np.load("ytrain_Classification2.npy")
X=X.reshape(X.shape[0],28,28,3)
# Load an image
image = X[1]

# Define the kernel size (must be an odd number)
kernel_size = 5
sigma=0.7

# Apply mean filtering
#filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size),sigma)
filtered_image = apply_contrast(image,0.3)
plt.imshow(image)
plt.show()

plt.imshow(filtered_image)
plt.show()