import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
def read_data():
    df=pd.read_excel(r"C:\Users\cdj\Desktop\Dmax1531.xlsx")
    x=df.iloc[:,0:3]
    y=df.iloc[:,-1:].values
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=2024)
    return x_train, x_test, y_train, y_test
def single_models():
    LR=LinearRegression()
    model_train(LR,'LinearRegression')
    ridge=Ridge(random_state=2024)
    model_train(ridge,'ridge')
    lasso=Lasso()
    model_train(lasso,'lasso')
    elasticNet = ElasticNet(random_state=2024)
    model_train(elasticNet,'elasticNet')
    bayesianRidge=BayesianRidge()
    model_train(bayesianRidge,'bayesianRidge')
    svr=SVR()
    model_train(svr,'svr')
    KNN= KNeighborsRegressor()
    model_train(KNN,'KNN')
    LGBM=lgb.LGBMRegressor(random_state=2024)
    model_train(LGBM,'Lightgbm')
    XGB = XGBRegressor()
    model_train(XGB,'XGBoost')
    DT=DecisionTreeRegressor(random_state=2024)
    model_train(DT,'DT')
    ET=ExtraTreesRegressor(random_state=2024)
    model_train(ET,'ET')
    GBDT=GradientBoostingRegressor(random_state=2024)
    model_train(GBDT,'GradientBoosting')
    RF = RandomForestRegressor(random_state=2024)
    model_train(RF,'RandomForest')
def model_train(model, str):
    print('--------------' +str+ '---------------------------------------')
    x_train, x_test, y_train, y_test=read_data()
    y_train=y_train.ravel()
    print('-----------Train result--------------------------------------------')
    model=model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    train_R2=model.score(x_train,y_train)
    train_RMSE = mean_squared_error(y_train,train_pred)**0.5
    cross_val__score=cross_val_score(model,x_train,y_train,cv=10,scoring="neg_mean_squared_error",n_jobs=-1)
    print("train_R2:",train_R2)
    print("train_RMSE:",train_RMSE)
    print("cross_val__score:",cross_val__score)
    print('-----------Test result---------------------------------------------')
    test_pred=model.predict(x_test)
    test_R2=model.score(x_test,y_test)
    test_RMSE = mean_squared_error(y_test,test_pred)**0.5
    print("test_R2:",test_R2)
    print("test_RMSE:",test_RMSE)
    return
single_models()