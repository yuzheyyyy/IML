import numpy as np
import read_data
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt


#read data
data,lable=read_data.read_data()
data=np.array(data)
lable=np.array(lable)

#non_linear
data_nonlinear=np.hstack((data,np.square(data),np.exp(data),np.cos(data),np.ones((np.size(data,0),1))))

##normalize
#data_norm=preprocessing.scale(data_nonlinear)

#ridge regression parameter
ridge_lambda=np.array([0.01,0.1,1,10,100])


#RMSE setting
RMSE_lam=np.zeros(np.size(ridge_lambda))

#cross_validation
kf=KFold(n_splits=10)
for index,lam in enumerate(ridge_lambda):
    regr=Ridge(lam)
    RMSE=[]
    for train_inx,test_ind in kf.split(data_nonlinear):
        regr.fit(data_nonlinear[train_inx,:],lable[train_inx])
        rms = sqrt(mean_squared_error(lable[test_ind], regr.predict(data_nonlinear[test_ind])))
        RMSE.append(rms)
    RMSE_lam[index]=np.mean(np.array(RMSE))

print(RMSE_lam)