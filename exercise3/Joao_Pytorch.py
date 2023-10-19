import numpy as np
from sklearn.model_selection import train_test_split
from utils import self_augmentation_rotate_flip, self_augmentation_shift,self_augmentation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.utils import resample
# from imblearn.over_sampling import SMOTE

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(FFNN, self).__init__()

        self.hidden_layers = nn.ModuleList()
        prev_layer_size = input_size

        for layer_size in hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size
        
        self.output_layer = nn.Linear(prev_layer_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.sigmoid(layer(x))

        out = F.softmax(self.output_layer(x))
        return out

class FFNNDataset(Dataset):
    def __init__(self, X_array,y_array):
        self.X=torch.tensor(X_array,dtype=torch.float32)
        self.y=torch.tensor(y_array,dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel 
        # padding='same'
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding='same') #shape = 6,26,26
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  #shape = 16,11,11
        # an affine operation: y = Wx + b
        self.bn2 = nn.BatchNorm2d(20)
        
        #self.fc1 = nn.Linear(30 * 6 * 6,64 )  # 5*5 from image dimension
        self.fc1 = nn.Linear(20 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout=nn.Dropout(0.4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) #shape = 6,13,13
        #print(x.shape)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #shape=16,5,5
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        #x = F.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x),dim=1) #Apply by rows
        return x

class CNNDataset(Dataset):
    def __init__(self, X_array,y_array,mode=None,augmentation=False):
        self.X=torch.tensor(X_array,dtype=torch.float32).reshape(X_array.shape[0],3,28,28)
        #self.X = torch.mean(self.X, dim=1).unsqueeze(1)
        self.y=torch.tensor(y_array,dtype=torch.float32).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

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
    print(f"Balanced Accuracy: {(100*balanced_acc):>0.1f}%, Recall:{(100*recall):>0.1f}%, Specificity:{(100*specificity):>0.1f}%,  Avg loss: {test_loss:>8f} \n")
    return balanced_acc

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
    loss=loss.item()
    # print(f"Training loss: {loss:>7f}")
    return loss

def preprocessing(X_array, y_array, technique: str):
    if technique == 'oversampling':
        # sm = SMOTE(random_state=42)
        # X_array,y_array=sm.fit_resample(X_array,y_array)
        print("Enfim")

    elif technique == 'undersampling':
        maj_class = np.where(y == 0)[0]
        min_class = np.where(y == 1)[0]
        maj_undersampled = resample(maj_class, n_samples=len(min_class), random_state=42)
        X_array = X_array[np.concatenate([min_class, maj_undersampled])]
        y_array = y_array[np.concatenate([min_class, maj_undersampled])]

    elif technique == 'augmentation':
        X_array,y_array=self_augmentation_rotate_flip(X_array,y_array)

        #X_array,y_array=self_augmentation_shift(X_array,y_array)
        #X_array,y_array=self_augmentation(X_array,y_array)

    return X_array, y_array

if __name__ == "__main__":

    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = np.load("Xtrain_Classification1.npy")
    y = np.load("ytrain_Classification1.npy")

    label_0_count = np.sum(y == 0)
    label_1_count = np.sum(y == 1) 

    prop = label_0_count / (label_0_count + label_1_count)

    print(prop)

    print(X.shape)

    #Split dataset into train, val and test. (0.6, 0.2, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_ftrain, y_ftrain = X_train, y_train #Final training dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

    # n_samples_train, n_features = X_train.shape # 3752, 2352

    #Pre-processing
    X_train, y_train = preprocessing(X_train, y_train, technique='augmentation')
    X_ftrain, y_ftrain = preprocessing(X_ftrain, y_ftrain, technique='augmentation')
    weights = torch.tensor([0.5, 0.5])
    # weights = torch.tensor([float(1-prop), float(prop)])

    saturation_factor = -1.0 
    contrast_factor = 0.0 

    




    # Model selection
    model = CNN()

    #Model hyperparameters selection
    learning_rate = 1e-4
    batch_size = 256
    epochs = 15
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    #loss_fn = nn.CrossEntropyLoss() NNLoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3) # Good
    # optimizer = optim.SGD(model.parameters(), lr=1e-4) # Melhor

    train_dataloader = DataLoader(CNNDataset(X_train,y_train), batch_size=batch_size,shuffle=True)
    val_dataloader = DataLoader(CNNDataset(X_val,y_val), batch_size=batch_size,shuffle=True)
    ftrain_dataloader = DataLoader(CNNDataset(X_ftrain,y_ftrain), batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(CNNDataset(X_test,y_test), batch_size=batch_size,shuffle=True)
    
    
    # Definir parametros da rede neuronal
    # hidden_layer_sizes = [128, 32]
    # output_size = 2 
    # model = FFNN(n_features, hidden_layer_sizes, output_size)

    #Train Model
    for epoch in range(epochs): 
        avg_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        # print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')

    print("Validation Results:")
    test_loop(val_dataloader,model,loss_fn)

    # Model selection
    model = CNN()

    #Model hyperparameters selection
    learning_rate = 1e-4
    batch_size = 256
    epochs = 15
    loss_fn = nn.CrossEntropyLoss(weight=weights) #NNLoss
    #loss_fn = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.01)

    # Treinar o modelo com o dataset train + val
    for epoch in range(epochs): 
        avg_loss = train_loop(ftrain_dataloader, model, loss_fn, optimizer)
        # print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')

    #Resultados do teste
    print("Test Results:")
    test_loop(test_dataloader,model,loss_fn)
