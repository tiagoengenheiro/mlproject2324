from cnn_py_torch import *
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
X=np.load("Xtrain_Classification2.npy")
y=np.load("ytrain_Classification2.npy")
X_test=np.load("Xtest_Classification2.npy")
model=CNN(dropout=0.4)

model.load_state_dict(torch.load("./self_augmentation_rotation_flip_model_weights/model_b_acc85.05_epochs29_loss0.436_split_0"))
print(model)
model.eval()
X_test=torch.tensor(X_test,dtype=torch.float32).reshape(X_test.shape[0],3,28,28) 
X_test/=255.0 #add normalization
with torch.no_grad():
    pred = model(X_test)
    pred=torch.argmax(pred,dim=1).cpu()
pred=np.array(pred,dtype=np.float64)
np.save("ytest_Classification2.npy",pred)
y_test=np.load("ytest_Classification2.npy")
print(y_test.shape)


#Test if we get the 0.85%
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for _, (train_index, test_index) in enumerate(skf.split(X, y)):
    test_dataloader = DataLoader(FFNDataset(X[test_index],y[test_index],mode="test"),batch_size=X[test_index].shape[0],shuffle=False)
    loss_fn = nn.NLLLoss() #NLLLoss
    avg_loss,balanced_acc=test_loop(test_dataloader,model,loss_fn,'cpu')
    print(balanced_acc)
    break