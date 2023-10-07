import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge,RANSACRegressor
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import RobustScaler
from utils import get_best_model
X_train_init=np.load("X_train_regression2.npy") #shape -> (100, 4)
y_train_init=np.load("y_train_regression2.npy") #shape -> (100, 1)

n_examples,n_features=X_train_init.shape

MAD=np.median(np.abs(y_train_init-np.median(y_train_init))) #0.9395742394919429
MAE=np.mean(np.abs(y_train_init-np.mean(y_train_init)))
min_mse=(np.inf,np.inf)
model1,model2=None,None
best_n_samples=0
th=0
for threshold in [MAD]:
    print("Threshold:", threshold)
    print("How many values above threshold:",len(y_train_init[y_train_init>threshold]))
    max_n_samples=len(y_train_init[y_train_init>threshold])
    for n_samples in range(1,10,1):
        reg = RANSACRegressor(random_state=0,min_samples=n_samples,max_trials=200,residual_threshold=threshold,loss='absolute_error',stop_probability=1.0)
        reg.fit(X_train_init,y_train_init)
        #print("Number of inliers",len(reg.inlier_mask_[reg.inlier_mask_==True]))
        model1,mse1=get_best_model(X_train_init[reg.inlier_mask_],y_train_init[reg.inlier_mask_])
        model2,mse2=get_best_model(X_train_init[~reg.inlier_mask_],y_train_init[~reg.inlier_mask_])
        print(n_samples,"MSE1",mse1,"MSE2",mse2)
        if (mse1+mse2) < sum(min_mse):
            min_mse=(mse1,mse2)
            best_n_samples=n_samples
            th=threshold
    print("Best MS1",min_mse[0],"Best MS2",min_mse[1])
    print("\n")
print("Best Threshold",threshold)
print("Best number of samples",best_n_samples)
print(model1,min_mse[0])
print(model2,min_mse[1])

reg = RANSACRegressor(random_state=0,min_samples=best_n_samples,residual_threshold=threshold,loss='absolute_error')
reg.fit(X_train_init,y_train_init)
print("Number of inliers",len(reg.inlier_mask_[reg.inlier_mask_==True]))
print("List of Outliers indexes",[i for i,x in enumerate(reg.inlier_mask_) if x==False])

model1,mse1=get_best_model(X_train_init[reg.inlier_mask_],y_train_init[reg.inlier_mask_])
model2,mse2=get_best_model(X_train_init[~reg.inlier_mask_],y_train_init[~reg.inlier_mask_])
#results=np.hstack((RS.inverse_transform(model1.predict(X_train_init)),RS.inverse_transform(model2.predict(X_train_init))))
results=np.hstack((model1.predict(X_train_init),model2.predict(X_train_init).reshape(n_examples,1)))
results = np.square(results-y_train_init)
results = np.min(results,axis=1)
results = np.sum(results,axis=0)
print("SSE:",results)


X_test_init=np.load("X_test_regression2.npy")
print(X_test_init.shape)

n_examples,n_features=X_test_init.shape

y_pred_1=model1.predict(X_test_init).reshape(n_examples,1)
y_pred_2=model2.predict(X_test_init).reshape(n_examples,1)
y_pred=np.hstack((y_pred_1,y_pred_2))
print(y_pred.shape)
np.save("y_test_regression2.npy",y_pred)
y_pred=np.load("y_test_regression2.npy")
print(y_pred.shape)