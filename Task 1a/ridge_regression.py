import numpy as np
import read_data
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from math import sqrt


#read data
data,lable=read_data.read_data()

#normalize
data_norm=preprocessing.scale(data)

#ridge regression parameter
ridge_lambda=np.array([0.01,0.1,1,10,100])

#cross_validation
lam_loss=np.zeros(5)
for index,lam in enumerate(ridge_lambda):
    regr=Ridge(lam)
    loss=np.array(-cross_val_score(regr,data_norm,lable,cv=10,scoring='neg_mean_squared_error'))
    RMSE=np.mean(np.sqrt(loss))
    lam_loss[index]=RMSE

np.savetxt('RMSE.csv',lam_loss,delimiter='\r\n')