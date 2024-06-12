import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold,cross_validate,train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import hyperopt
from hyperopt import hp,fmin,tpe,Trials,partial
from hyperopt.early_stop import no_progress_loss
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import KFold,cross_validate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated")
def read_data():
    df = pd.read_excel(r"C:\Users\cdj\Desktop\smogn.xlsx")
    x=df.iloc[:,:3]
    y=df.iloc[:,-1:].values.ravel()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2024)
    return x_train,x_test,y_train,y_test
def RFforest_model(val):
    x_train,x_test,y_train,y_test=read_data()
    reg=RandomForestRegressor(random_state=2024,n_jobs=-1).fit(x_train,y_train)
    cv=KFold(n_splits=10,shuffle=True, random_state=2024)
    CVresult=cross_validate(reg,x_train,y_train,cv=cv,n_jobs=-1,return_train_score=True)
    print("RF")
    print("训练集R2：%f"%CVresult["train_score"].mean())
    print("验证集R2：%f"%CVresult["test_score"].mean())
    print("测试集R2：%f"%reg.score(x_test,y_test))
    def hypeopt_objective(params):
        reg=RFR(n_estimators=int(params["n_estimators"])
            ,max_depth=int(params["max_depth"])
            ,max_features=params["max_features"]
            ,min_samples_split=int(params["min_samples_split"])
            ,random_state=2024
            ,verbose=False
            ,n_jobs=-1
            )
        cv = KFold(n_splits=10, shuffle=True, random_state=2024)
        validation_loss = cross_validate(reg
                                         , x_train, y_train
                                         , cv=cv
                                         , verbose=False
                                         , n_jobs=-1
                                         , error_score='raise'
                                         )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                    'n_estimators':hp.quniform("n_estimators",50,90,1)
                    ,'max_depth':hp.quniform("max_depth",15,35,1)
                    ,"max_features": hp.uniform("max_features",0.6,0.9)
                    ,"min_samples_split":hp.quniform("min_samples_split",2,5,1)
                    }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(50)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn
                        )
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
def KNN_model(val):
    x_train,x_test,y_train,y_test=read_data()
    KNN=KNeighborsRegressor(n_jobs=-1).fit(x_train,y_train)
    cv=KFold(n_splits=10,shuffle=True, random_state=2024)
    CVresult=cross_validate(KNN,x_train,y_train,cv=cv,n_jobs=-1,return_train_score=True)
    print("KNN")
    print("训练集R2：%f"%CVresult["train_score"].mean())
    print("验证集R2：%f"%CVresult["test_score"].mean())
    print("测试集R2：%f"%KNN.fit(x_train,y_train).score(x_test,y_test))
    def hypeopt_objective(params):
        reg=KNeighborsRegressor(n_neighbors=int(params["n_neighbors"])
                                ,leaf_size=int(params["leaf_size"])
                                ,algorithm=params["algorithm"]
                                ,weights=params["weights"]
                                ,n_jobs=-1)
        cv = KFold(n_splits=10, shuffle=True, random_state=2024)
        validation_loss = cross_validate(reg
                                         , x_train, y_train
                                         , cv=cv
                                         , verbose=False
                                         , n_jobs=-1
                                         , error_score='raise'
                                         )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'n_neighbors':hp.quniform("n_neighbors",1,10,1)
                        ,'leaf_size':hp.quniform("leaf_size",10,60,1)
                        ,'algorithm':hp.choice("algorithm",['auto', 'ball_tree', 'kd_tree', 'brute'])
                        ,'weights': hp.choice('weights', ['uniform', 'distance'])
                        }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(50)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn
                        )
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
def ET_model(val):
    x_train,x_test,y_train,y_test=read_data()
    reg=ExtraTreesRegressor(random_state=2024,n_jobs=-1).fit(x_train,y_train)
    cv=KFold(n_splits=10,shuffle=True, random_state=2024)
    CVresult=cross_validate(reg,x_train,y_train,cv=cv,n_jobs=-1,return_train_score=True)
    print("ET")
    print("训练集R2：%f"%CVresult["train_score"].mean())
    print("验证集R2：%f"%CVresult["test_score"].mean())
    print("测试集R2：%f"%reg.score(x_test,y_test))
    param_grid_simple={ "n_estimators":hp.quniform("n_estimators",400,600,1)
                       ,"max_features": hp.uniform("max_features",0.7,0.99)#下降
                       ,"max_depth":hp.quniform("max_depth",10,40,1)
                        }
    def hypeopt_objective(params):
        reg=ExtraTreesRegressor(n_estimators=int(params["n_estimators"])
                ,max_features=params["max_features"]
                ,max_depth=int(params["max_depth"])
                ,random_state=2024
                ,verbose=False
                ,n_jobs=-1
                )
        cv = KFold(n_splits=10, shuffle=True, random_state=2024)
        validation_loss = cross_validate(reg
                                         , x_train, y_train
                                         , cv=cv
                                         , verbose=False
                                         , n_jobs=-1
                                         , error_score='raise'
                                         )
        return -np.mean(validation_loss["test_score"])
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(50)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn
                        )
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
def XGB_model(val):
    x_train,x_test,y_train,y_test=read_data()
    reg=xgb.XGBRegressor(random_state=2024,n_jobs=-1).fit(x_train,y_train)
    cv=KFold(n_splits=10,shuffle=True, random_state=2024)
    CVresult=cross_validate(reg,x_train,y_train,cv=cv,n_jobs=-1,return_train_score=True)
    print("XGB")
    print("训练集R2：%f"%CVresult["train_score"].mean())
    print("验证集R2：%f"%CVresult["test_score"].mean())
    print("测试集R2：%f"%reg.score(x_test,y_test))
    param_grid_simple={ "n_estimators":hp.quniform("n_estimators",400,600,1)
                       ,"max_features": hp.uniform("max_features",0.7,0.99)
                       ,"max_depth":hp.quniform("max_depth",10,40,1)
                        }
    def hypeopt_objective(params):
        reg=xgb.XGBRegressor(n_estimators=int(params["n_estimators"])
                ,learning_rate=params["learning_rate"]
                ,max_depth=int(params["max_depth"])
                ,subsample=params["subsample"]
                ,colsample_bytree=params["colsample_bytree"]
                ,reg_alpha=params["reg_alpha"]
                ,reg_lambda=params["reg_lambda"]
                ,random_state=2024
                ,n_jobs=-1
                )
        cv = KFold(n_splits=10, shuffle=True, random_state=2024)
        validation_loss = cross_validate(reg
                                         , x_train, y_train
                                         , cv=cv
                                         , verbose=False
                                         , n_jobs=-1
                                         , error_score='raise'
                                         )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'n_estimators':hp.quniform("n_estimators",250,350,1)
                        ,'learning_rate':hp.uniform("learning_rate",0.1,0.4)
                        ,'max_depth':hp.quniform("max_depth",5,30,1)
                        ,"subsample": hp.uniform("subsample",0.4,0.9)
                        ,"colsample_bytree":hp.uniform("colsample_bytree",0.5,1.0)
                        ,"reg_alpha": hp.uniform("reg_alpha",0,0.1)
                        ,"reg_lambda": hp.uniform("reg_lambda",1.5,2.5)
                        }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(50)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn
                        )
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
def LGBM_model(val):
    x_train,x_test,y_train,y_test=read_data()
    reg=lgb.LGBMRegressor(random_state=2024,n_jobs=-1).fit(x_train,y_train)
    cv=KFold(n_splits=10,shuffle=True, random_state=2024)
    CVresult=cross_validate(reg,x_train,y_train,cv=cv,n_jobs=-1,return_train_score=True)
    print("LGBM")
    print("训练集R2：%f"%CVresult["train_score"].mean())
    print("验证集R2：%f"%(CVresult["test_score"].mean()))
    print("测试集R2：%f"%reg.score(x_test,y_test))
    param_grid_simple={ "n_estimators":hp.quniform("n_estimators",400,600,1)
                       ,"max_features": hp.uniform("max_features",0.7,0.99)
                       ,"max_depth":hp.quniform("max_depth",10,40,1)
                        }
    def hypeopt_objective(params):
        reg=lgb.LGBMRegressor(num_leaves=int(params["num_leaves"])
                ,max_depth=int(params["max_depth"])
                ,min_child_samples=int(params["min_child_samples"])
                ,learning_rate=params["learning_rate"]
                ,n_estimators=int(params["n_estimators"])
                ,min_child_weight=params["min_child_weight"]
                ,subsample=params["subsample"]
                ,reg_lambda=params["reg_lambda"]
                ,random_state=2024
                ,n_jobs=-1
                )
        cv = KFold(n_splits=10, shuffle=True, random_state=2024)
        validation_loss = cross_validate(reg
                                         , x_train, y_train
                                         , cv=cv
                                         , verbose=False
                                         , n_jobs=-1
                                         , error_score='raise'
                                         )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'num_leaves':hp.quniform("num_leaves",25,40,1)
                        ,"learning_rate": hp.uniform("learning_rate",0.2,0.4)
                        ,'max_depth':hp.quniform("max_depth",5,20,1)
                        ,'min_child_samples':hp.quniform("min_child_samples",1,5,1)
                        ,'n_estimators':hp.quniform("n_estimators",100,250,1)
                        ,'min_child_weight': hp.uniform("min_child_weight",0.001,0.2)#数值不对
                        ,'subsample': hp.uniform("subsample",0.2,0.9)
                        ,'reg_lambda':hp.uniform("reg_lambda",0.1,0.6)
                                            }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(50)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn
                        )
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
def GBDT_model(val):
    x_train,x_test,y_train,y_test=read_data()
    reg=GradientBoostingRegressor(random_state=2024).fit(x_train,y_train)
    cv=KFold(n_splits=10,shuffle=True, random_state=2024)
    CVresult=cross_validate(reg,x_train,y_train,cv=cv,n_jobs=-1,return_train_score=True)
    print("GBDT")
    print("训练集R2：%f"%CVresult["train_score"].mean())
    print("验证集R2：%f"%(CVresult["test_score"].mean()))
    print("测试集R2：%f"%reg.score(x_test,y_test))
    def hypeopt_objective(params):
        reg=GradientBoostingRegressor(min_samples_leaf=int(params["min_samples_leaf"])
                ,max_depth=int(params["max_depth"])
                ,learning_rate=params["learning_rate"]
                ,n_estimators=int(params["n_estimators"])
                ,random_state=2024
                )
        cv = KFold(n_splits=10, shuffle=True, random_state=2024)
        validation_loss = cross_validate(reg
                                         , x_train, y_train
                                         , cv=cv
                                         , verbose=False
                                         , n_jobs=-1
                                         , error_score='raise'
                                         )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'min_samples_leaf':hp.quniform("min_samples_leaf",3,7,1)
                        ,"learning_rate": hp.uniform("learning_rate",0.001,0.3)
                        ,'max_depth':hp.quniform("max_depth",5,15,1)
                        ,'n_estimators':hp.quniform("n_estimators",200,600,1)
     }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(50)#It does not decrease after 50 successive iterations
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn
                        )
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
val=100#The number of iterations required
XGB_model(val)
RFforest_model(val)
KNN_model(val)
ET_model(val)
LGBM_model(val)
GBDT_model(val)