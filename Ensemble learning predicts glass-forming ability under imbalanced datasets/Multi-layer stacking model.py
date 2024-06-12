import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,cross_validate,train_test_split,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
df=pd.read_excel(r"C:\Users\cdj\Desktop\smogn.xlsx")
x=df.iloc[:,:3]
y=df.iloc[:,-1:].values
y=y.ravel()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2024)
svr=SVR()
LR=LinearRegression()
KNN= KNeighborsRegressor(algorithm='ball_tree'
                         ,leaf_size=23
                         ,n_neighbors=2
                         ,weights='distance'
                         ,n_jobs=-1).fit(x_train,y_train)
LGBM=lgb.LGBMRegressor(learning_rate= 0.256967437142703
                       ,max_depth=9
                       ,min_child_samples=1
                       ,min_child_weight=0.17375982523233643
                       ,n_estimators=155
                       ,num_leaves=35
                       ,reg_lambda=0.20664218563485248
                       ,subsample=0.48890220331312
                       ,random_state=2024).fit(x_train,y_train)
XGB=XGBRegressor(colsample_bytree=0.9695972795312625,
                  learning_rate=0.01107378739738763
                  ,max_depth=30
                  ,n_estimators=386
                  ,reg_alpha=0.0579535095359525
                  ,reg_lambda=1.4000269267347374
                  ,subsample=0.5990161291866896
                  ,random_state=2024
                  ,n_jobs=-1 ).fit(x_train,y_train)
RF=RandomForestRegressor(max_depth=17
                         ,max_features=0.6480711253210274
                         ,min_samples_split=2
                         ,n_estimators=69
                         ,random_state=2024).fit(x_train,y_train)
ET=ExtraTreesRegressor(max_depth=15
                       ,n_estimators=566
                       ,max_features=0.802111399665188
                       ,random_state=2024).fit(x_train,y_train)
GBDT=GradientBoostingRegressor(learning_rate=0.07664064490807247
                               ,max_depth=10
                               ,min_samples_leaf=7
                               ,n_estimators=263
                               ,random_state=2024).fit(x_train,y_train)
stack_gen21=StackingCVRegressor(regressors=(KNN, RF, GBDT, LGBM, ET)
                                                ,meta_regressor=LGBM
                                                ,shuffle=True
                                                ,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(KNN, KNN, LGBM, KNN, RF)
                                            ,meta_regressor=XGB
                                            ,shuffle=True
                                            ,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(ET, ET, XGB, RF, LGBM)
                                            ,meta_regressor=KNN
                                            ,shuffle=True
                                            ,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                           ,meta_regressor=LR
                                           ,shuffle=True
                                           ,n_jobs=-1
                                           ,random_state=2024).fit(x_train,y_train)
train_score=stack_gen3.score(x_train,y_train)
test_score=stack_gen3.score(x_test,y_test)
print("train_score,test_score,cross_score：{},{}".format(train_score,test_score))