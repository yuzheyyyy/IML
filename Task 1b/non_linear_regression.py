import numpy as np
import read_data
from sklearn.svm import LinearSVR

#read data
data,lable=read_data.read_data()
data=np.array(data)
lable=np.array(lable)

#non_linear
data_nonlinear=np.hstack((data,np.square(data),np.exp(data),np.cos(data),np.ones((np.size(data,0),1))))

#
regr=LinearSVR(C=.01)
regr.fit(data_nonlinear,lable)
print(regr.coef_)

#save data
np.savetxt("model_coefficient.csv",regr.coef_,delimiter='\r\n')

##normalize
#data_norm=preprocessing.scale(data_nonlinear)
#lable_norm=preprocessing.scale(lable)

##ridge regression parameter
#ridge_lambda=np.linspace(0.01,0.1,100)
#
##find minimum RMSE for lambda
#min_RMSE=math.inf
#RMSE_coef=np.zeros(21)
#
##RMSE setting
#lam_loss=np.zeros(np.size(ridge_lambda))

##cross_validation
#for ind,lam in enumerate(ridge_lambda):
#    regr=Lasso(lam)
#    loss=-cross_val_score(regr,data_norm,lable,cv=10,scoring='neg_mean_squared_error')
#    lam_loss[ind]=np.mean(np.sqrt(loss))
#
#lam=0.02
#model=Lasso(lam)
#model.fit(data_nonlinear,lable)
#print(model.coef_)
#print(lam_loss[np.where(ridge_lambda==lam)])
#print(ridge_lambda)
#
##plot
#plt.plot(ridge_lambda,lam_loss)




