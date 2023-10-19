import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN, KMeansSMOTE,BorderlineSMOTE

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
 #https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy") #.reshape(X.shape[0],1)

#Pre processing:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #0.8 - 0.2 

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

print("Split: Y","N_examples:",len(y),"Class 0:",len(y[y==0]),"Class 1:",len(y[y==1]),"Ratio:",len(y[y==0])/len(y[y==1]))
print("Split: Y_train",len(y_train),"Class 0:",len(y_train[y_train==0]),"Class 1:",len(y_train[y_train==1]),"Ratio:",len(y_train[y_train==0])/len(y_train[y_train==1]))
print("Split: Y_test",len(y_test),"Class 0:",len(y_test[y_test==0]),"Class 1:",len(y_test[y_test==1]),"Ratio:",len(y_test[y_test==0])/len(y_test[y_test==1]))
# print("Split: Y_val",len(y_val),"Class 0:",len(y_val[y_val==0]),"Class 1:",len(y_val[y_val==1]),"Ratio:",len(y_val[y_val==0])/len(y_val[y_val==1]))

# _,n_features=X_train.shape
# print("N of features:",n_features)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel 
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5,padding='same') #shape = 6,26,26
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)  #shape = 16,11,11
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20 * 6 * 6,120 )  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 2)
        self.dropout=nn.Dropout(0.5) #Regularizacao

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) #shape = 6,13,13
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #shape=16,5,5
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.log_softmax(self.fc2(x),dim=1) #Apply by rows
        return x
    
class FFNDataset(Dataset):
    def __init__(self, X_array,y_array,mode=None,oversampling=False,augmentation=False,std=0.05,th=0.3):
        if mode=="train":
            if oversampling:
                print("Using Oversampling")
                sm = SMOTE(random_state=42)
                X_array,y_array=sm.fit_resample(X_array,y_array)
            elif augmentation:
                print("Using Augmentation")
                X_array,y_array=self_augmentation(X_array,y_array)
                print(X_array.shape)
        self.X=torch.tensor(X_array,dtype=torch.float32).reshape(X_array.shape[0],3,28,28)
        self.y=torch.tensor(y_array,dtype=torch.float32).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    num_batches = len(dataloader)
    train_loss=0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss+=loss.item()
        # Backpropagation
        loss.backward() #computed the loss for every parameter
        optimizer.step() #updates the parameters
        optimizer.zero_grad() #resets the gradients
    train_loss /= num_batches
    print(f"Avg train loss: {train_loss:>7f}")
    return train_loss

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss=0
    TP,FP,FN,TN = 0,0,0,0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred=torch.argmax(pred,dim=1) #dim=1 is by row
            for i,y_hat in enumerate(pred):
                if y_hat==y[i]: #If the predicted is equivalent to the Gold than it's True
                    if y_hat==1: 
                        TP+=1
                    else:
                        TN+=1
                else: #If the predicted is not equivalent to the Gold than it's False
                    if y_hat==1: #Predicted as Positive but it's Negative
                        FP+=1
                    else: #Predicted as Negative but it's Positive
                        FN+=1
    recall=TP/(TP+FN)
    specificity=TN/(TN+FP)
    balanced_acc = 1/2*(recall+specificity)
    test_loss /= num_batches
    print(f"Recall:{(100*recall):>0.1f}%, Specificity:{(100*specificity):>0.1f}%, Balanced Accuracy: {(100*balanced_acc):>0.1f}%,  Avg loss: {test_loss:>8f} \n")
    return test_loss,recall,specificity,balanced_acc

#flipping images vertically or creating mirrored images
model=CNN()
learning_rate = 1e-3
batch_size = 128
epochs = 20
weight_decay=0.01
loss_fn = nn.NLLLoss() #NLLoss
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
std=0.05
th=1.0
#Loads the data in a dataloader to control the batch and pre-processing easier
train_dataloader = DataLoader(FFNDataset(X_train,y_train,mode="train",oversampling=False,augmentation=True,std=std,th=th), batch_size=batch_size,shuffle=True)
# val_dataloader = DataLoader(FFNDataset(X_val,y_val), batch_size=batch_size,shuffle=True)



print("Test Results on X_test")
test_dataloader = DataLoader(FFNDataset(X_test,y_test), batch_size=batch_size)

train_loss_list=[]
test_loss_list=[]
b_acc_list=[]
specifity_list=[]
recall_list=[]

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss_list.append(train_loop(train_dataloader, model, loss_fn, optimizer))
    print(f"Validation Results:")
    avg_loss,recall,specificity,balanced_acc=test_loop(test_dataloader, model, loss_fn)
    test_loss_list.append(avg_loss)
    b_acc_list.append(round(100*balanced_acc,2))
    specifity_list.append(round(100*specificity,2))
    recall_list.append(round(100*recall,2))
print("Done Traning!")

plt.figure(figsize=(10, 6))
plt.plot(train_loss_list, label='train_loss')
plt.plot(test_loss_list,label='test_loss')
plt.legend()
#augmentation=f"shift_n_2_left_shift_axis_1"
#augmentation=f"shift_n_2_flip_axis_0_merged_with_saturation_std_{std}_th_{th}"
plt.savefig(f"graphs/lr_{learning_rate}_weight_decay_{weight_decay}_{augmentation}.png")
plt.figure(figsize=(10, 6)) 
plt.plot(b_acc_list, label='Balanced Accuracy')
plt.plot(specifity_list,label='Specifity')
plt.plot(recall_list,label='Recall')
plt.legend()
plt.savefig(f"graphs/max_bacc_{max(b_acc_list)}_loss_{round(test_loss_list[b_acc_list.index(max(b_acc_list))],3)}_epochs_{b_acc_list.index(max(b_acc_list))+1}_lr_{learning_rate}_weigth_decay_{weight_decay}_{augmentation}.png")