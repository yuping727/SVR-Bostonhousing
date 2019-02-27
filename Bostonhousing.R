rm(list=ls())

# Boston housing data 
install.packages("mlbench")
library(mlbench)
data(BostonHousing)
housing<- BostonHousing
str(housing)
head(housing)
dim(housing)
sum(is.na(housing)) # no missing value 

select.housing <- housing[,c("medv", "rm", "tax", "indus", "lstat", "ptratio", "nox")] 
dim(select.housing)
# create model matrix x 
x<-model.matrix(medv~ rm+tax+indus+poly(lstat,3)+ptratio+nox, data=select.housing)[,-1]   # -1 means no intercept 
head(x)
dim(x)
y=select.housing$medv

# train and test dataset
set.seed(1)
train=sample(1:nrow(x),nrow(x)/2)
test=(-train)
y.test=y[test]

# now use cv to find the best lambda
library(glmnet)
cv<- cv.glmnet(x,y,type.measure = "mse",alpha=0)   #alpha=0 is for ridge, =1 is lasso
plot(cv$lambda,cv$cvm)    # cvm is mean cross-validated error 
bestlam<- cv$lambda.min     # select the value of lambda from leave-10-out cross validation

# fit ridge regression model on training set
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=bestlam)
ridge.pred <- predict(ridge.mod, s=bestlam, newx=x[test,])

# test RMSE 
# define rmse function
error= ridge.pred-y.test

rmse=function(error)
{
  sqrt(mean(error^2))
}

RidgeRMSE= rmse(error)

print(RidgeRMSE)  # 4.840246






rm(list=ls())
# SVR with RBF kernel 

library(mlbench)
data(BostonHousing)
housing<- BostonHousing
select.housing <- housing[,c("medv", "rm", "tax", "indus", "lstat", "ptratio", "nox")] 
plot(select.housing)

x<-model.matrix(medv~., data=select.housing)[,-1] 
y=select.housing$medv
# train and test dataset 
set.seed(1)
train=sample(1:nrow(x),nrow(x)/2)
test=(-train)
y.test=y[test]

library(e1071)
# hyperparameter optimization: perform a grid search 
tuneResult=tune(svm,y[train]~ x[train,],data=select.housing[train,],ranges=list(epsilon=seq(0,1,0.1),cost=2^(2:9))) 
print(tuneResult) # best parameter  epsilon 0.1, cost 512, MSE=6.041437
# draw the tuning graph
plot(tuneResult)

# try another grid search in a narrow range 
tuneResult=tune(svm,y[train]~ x[train,],data=select.housing[train,],ranges=list(epsilon=seq(0,0.2,0.01),cost=2^(2:9))) 
print(tuneResult)
plot(tuneResult)   # best parameter epsilon 0.06, cost 256, best performance MSE= 5.940939


# best model
tunedModel= tuneResult$best.model
summary(tunedModel)  # cost 256, gamma 0.1666667, epsilon 0.06 

# prediction
tunedModelY= predict(tunedModel,data=select.housing[test,])
error= y.test-tunedModelY

# define rmse function
rmse=function(error)
{
  sqrt(mean(error^2))
}

tunedModelRMSE= rmse(error)

print(tunedModelRMSE)  # 12.57449






# kernlab 
library(kernlab)
model_kernlab= ksvm(y[train]~ x[train,],data=select.housing[train,],scaled=T,type="eps-svr",kernel="rbfdot", kpar="automatic",
           epsilon=0.06,cost=256)
#prediction
kernlabY= predict(object=model_kernlab,newdata=x[test,])
error_kernlab= y.test-kernlabY
kernlab_RMSE= rmse(error_kernlab)
print(kernlab_RMSE)   # 4.08001




# ksvm() function 
library(kernlab)
model_kernlab= ksvm(y[train]~ x[train,],data=select.housing[train,],scaled=T,type="eps-svr",kernel="rbfdot", kpar="automatic",
                    epsilon=0.06,cost=256,gamma=0.1666667)
#prediction
kernlabY= predict(object=model_kernlab,newdata=x[test,])
error_kernlab= y.test-kernlabY

# define rmse function
rmse=function(error)
{
  sqrt(mean(error^2))
}

kernlab_RMSE= rmse(error_kernlab)
print(kernlab_RMSE)   # 4.025771











