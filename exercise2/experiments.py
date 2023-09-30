import numpy as np
import matplotlib.pyplot as plt
a_pos=1+np.random.randn(51,1)
a_neg=-1+np.random.randn(49,1)
print(np.mean(a_pos),np.std(a_pos))
print(np.mean(a_neg),np.std(a_neg))
#a=np.vstack((a_pos,a_neg))
a=np.random.randn(100,1)
print(a.shape)

print(np.mean(a),np.std(a))
bins=20
plt.hist(a, bins=bins, edgecolor='k')  # You can adjust the number of bins as needed

# Add labels and a title
plt.xlabel('Variable Values')
plt.ylabel('Frequency')
plt.title('Histogram of Output')
plt.show()