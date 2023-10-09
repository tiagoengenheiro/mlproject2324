import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

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
        out = self.output_layer(x)
        return out


X = np.load("Xtrain_Classification1.npy")
y = np.load("ytrain_Classification1.npy").reshape(X.shape[0],1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #0.8 | 0.2 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.6 | 0.2 | 0.2

n_samples_train, n_features = X_train.shape # 3752, 2352 e o y: 3752, 1 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_train_tensor = y_train_tensor.squeeze().long()

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_val_tensor = y_val_tensor.squeeze().long()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
y_test_tensor = y_test_tensor.squeeze().long()


# Definir parametros da rede neuronal
hidden_layer_sizes = [128, 32]
output_size = 2 

model = FFNN(n_features, hidden_layer_sizes, output_size)

# Cross-Entropy porque estamos a utilizar 2+ neuroes na output layer
criterion = nn.CrossEntropyLoss()

# Especificar aqui o optimizer e a learning rate
# optimizer = optim.SGD(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Definir as epochs
epochs = 100

for epoch in range(epochs):
    outputs = model(X_train_tensor)
    print(outputs)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# Avaliar o modelo
with torch.no_grad():
    print(y_test_tensor.shape)
    predicted = model(X_test_tensor)
    
	# Get the highest result
    predicted_class = torch.argmax(predicted, dim=1)
    print(predicted_class.shape)
	
    accuracy = (predicted_class == y_test_tensor).sum().item() / y_test_tensor.size(0) * 100
    print(f'Accuracy: {accuracy:.2f}%')

