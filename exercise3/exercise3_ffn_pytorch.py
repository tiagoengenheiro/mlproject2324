import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy").reshape(X.shape[0],1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #0.8 - 0.2 

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
#0.6 Train and 0.2 Val and 0.2 Test
print("Split: Y","N_examples:",len(y),"Class 0:",len(y[y==0]),"Class 1:",len(y[y==1]),"Ratio:",len(y[y==0])/len(y[y==1]))
print("Split: Y_train",len(y_train),"Class 0:",len(y_train[y_train==0]),"Class 1:",len(y_train[y_train==1]),"Ratio:",len(y_train[y_train==0])/len(y_train[y_train==1]))
print("Split: Y_test",len(y_test),"Class 0:",len(y_test[y_test==0]),"Class 1:",len(y_test[y_test==1]),"Ratio:",len(y_test[y_test==0])/len(y_test[y_test==1]))
print("Split: Y_val",len(y_val),"Class 0:",len(y_val[y_val==0]),"Class 1:",len(y_val[y_val==1]),"Ratio:",len(y_val[y_val==0])/len(y_val[y_val==1]))

_,n_features=X_train.shape
print("N of features:",n_features)

class FFN(nn.Module):

    def __init__(self):
        super(FFN, self).__init__()
        self.fc1=nn.Linear(n_features, 256)
        self.fc2=nn.Linear(256, 1)
    def forward(self, x):
        x=F.tanh(self.fc1(x))
        x=F.sigmoid(self.fc2(x))
        return x
    
class FFNDataset(Dataset):
    def __init__(self, X_array,y_array):
        self.X=torch.tensor(X_array,dtype=torch.float32)
        self.y=torch.tensor(y_array,dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() #computed the loss for every parameter
        optimizer.step() #updates the parameters
        optimizer.zero_grad() #resets the gradients
    loss, current = loss.item(), (batch + 1) * len(X)
    print(f"loss: {loss:>7f}")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss=0
    TP,FP,FN,TN = 0,0,0,0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred=torch.round(pred) #round to 0
            for i,y_hat in enumerate(pred):
                if y_hat==y[i]:
                    if y_hat==1: 
                        TP+=1
                    else:
                        TN+=1
                else:
                    if y_hat==1: #Predicted as Positive but it's Negative
                        FP+=1
                    else: #Predicted as Negative but it's Positive
                        FN+=1
        print(TP,TN,FP,FN,size,TP+FP+FN+TN)
    balanced_acc =1/2*(TP/(TP+FN)+TN/(TN+FP))
    test_loss /= num_batches
    print(f"Test Error: \n Balanced Accuracy: {(100*balanced_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")


model=FFN()
learning_rate = 1e-3
batch_size = 64
epochs = 50
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(FFNDataset(X_train,y_train), batch_size=batch_size)
val_dataloader = DataLoader(FFNDataset(X_val,y_val), batch_size=batch_size)
test_loss, correct = 0, 0


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(val_dataloader, model, loss_fn)
print("Done Traning!")

val_dataloader = DataLoader(FFNDataset(X_test,y_test), batch_size=batch_size)
print("Test Results")
test_loop(val_dataloader,model,loss_fn)