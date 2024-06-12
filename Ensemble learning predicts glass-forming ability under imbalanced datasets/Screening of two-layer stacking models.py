import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
import hyperopt
from hyperopt import hp,fmin,tpe,Trials,partial
from hyperopt.early_stop import no_progress_loss
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="X has feature names")
df = pd.read_excel(r"C:\Users\cdj\Desktop\smogn.xlsx")
x=df.iloc[:,:3]
y=df.iloc[:,-1:].values.ravel()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=2024)
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
def RF_model(val):
    def hypeopt_objective(params):
        model1 = params["model1"]
        model2 = params["model2"]
        model3 = params["model3"]
        model4 = params["model4"]
        model5 = params["model5"]
    #model1
        if model1=='KNN':
            base_model1= KNN
        elif model1=='LGBM':
            base_model1=LGBM
        elif model1=='XGB':
            base_model1 =XGB
        elif model1=='ET':
            base_model1=ET
        elif model1=='GBDT':
            base_model1=GBDT
        elif model1=='RF':
            base_model1 =RF
    #model2
        if model2=='KNN':
            base_model2= KNN
        elif model2=='LGBM':
            base_model2=LGBM
        elif model2=='XGB':
            base_model2 = XGB
        elif model2=='ET':
            base_model2=ET
        elif model2=='GBDT':
            base_model2=GBDT
        elif model2=='RF':
            base_model2 = RF
    #model3
        if model3=='KNN':
            base_model3= KNN
        elif model3=='LGBM':
            base_model3=LGBM
        elif model3=='XGB':
            base_model3 = XGB
        elif model3=='ET':
            base_model3=ET
        elif model3=='GBDT':
            base_model3=GBDT
        elif model3=='RF':
            base_model3 =RF
    #model4
        if model4=='KNN':
            base_model4=KNN
        elif model4=='LGBM':
            base_model4=LGBM
        elif model4=='XGB':
            base_model4 =XGB
        elif model4=='ET':
            base_model4=ET
        elif model4=='GBDT':
            base_model4=GBDT
        elif model4=='RF':
            base_model4 =RF
    #model5
        if model5=='KNN':
            base_model5=KNN
        elif model5=='LGBM':
            base_model5=LGBM
        elif model5=='XGB':
            base_model5 =XGB
        elif model5=='ET':
            base_model5=ET
        elif model5=='GBDT':
            base_model5=GBDT
        elif model5=='RF':
            base_model5 =RF
        stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                          ,meta_regressor=RF
                                          ,shuffle=True
                                          ,n_jobs=-1
                                          ,random_state=2024)
        cv=KFold(n_splits=10,shuffle=True,random_state=2024)
        validation_loss=cross_validate(stack_gen
                                       ,x_train,y_train
                                       ,cv=cv
                                       ,verbose=False
                                       ,n_jobs=-1
                                       ,error_score='raise'
                                        )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'model1':hp.choice("model1",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model2':hp.choice("model2",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model3":hp.choice("model3",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model4":hp.choice("model4",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model5':hp.choice("model5",['KNN','LGBM','XGB','ET','GBDT','RF'])
                        }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(500)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn)
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
    model_list=['KNN','LGBM','XGB','ET','GBDT','RF']
    model1=model_list[params_best["model1"]]
    model2=model_list[params_best["model2"]]
    model3=model_list[params_best["model3"]]
    model4=model_list[params_best["model4"]]
    model5=model_list[params_best["model5"]]
    print("Metamodel is RF")
    print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
def GBDT_model(val):
    def hypeopt_objective(params):
        model1 = params["model1"]
        model2 = params["model2"]
        model3 = params["model3"]
        model4 = params["model4"]
        model5 = params["model5"]
    #model1
        if model1=='KNN':
            base_model1= KNN
        elif model1=='LGBM':
            base_model1=LGBM
        elif model1=='XGB':
            base_model1 =XGB
        elif model1=='ET':
            base_model1=ET
        elif model1=='GBDT':
            base_model1=GBDT
        elif model1=='RF':
            base_model1 =RF
    #model2
        if model2=='KNN':
            base_model2= KNN
        elif model2=='LGBM':
            base_model2=LGBM
        elif model2=='XGB':
            base_model2 = XGB
        elif model2=='ET':
            base_model2=ET
        elif model2=='GBDT':
            base_model2=GBDT
        elif model2=='RF':
            base_model2 = RF
    #model3
        if model3=='KNN':
            base_model3= KNN
        elif model3=='LGBM':
            base_model3=LGBM
        elif model3=='XGB':
            base_model3 = XGB
        elif model3=='ET':
            base_model3=ET
        elif model3=='GBDT':
            base_model3=GBDT
        elif model3=='RF':
            base_model3 =RF
    #model4
        if model4=='KNN':
            base_model4=KNN
        elif model4=='LGBM':
            base_model4=LGBM
        elif model4=='XGB':
            base_model4 =XGB
        elif model4=='ET':
            base_model4=ET
        elif model4=='GBDT':
            base_model4=GBDT
        elif model4=='RF':
            base_model4 =RF
    #model5
        if model5=='KNN':
            base_model5=KNN
        elif model5=='LGBM':
            base_model5=LGBM
        elif model5=='XGB':
            base_model5 =XGB
        elif model5=='ET':
            base_model5=ET
        elif model5=='GBDT':
            base_model5=GBDT
        elif model5=='RF':
            base_model5 =RF
        stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                          ,meta_regressor=GBDT
                                          ,shuffle=True
                                          ,n_jobs=-1
                                          ,random_state=2024)
        cv=KFold(n_splits=10,shuffle=True,random_state=2024)
        validation_loss=cross_validate(stack_gen
                                       ,x_train,y_train
                                       ,cv=cv
                                       ,verbose=False
                                       ,n_jobs=-1
                                       ,error_score='raise'
                                        )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'model1':hp.choice("model1",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model2':hp.choice("model2",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model3":hp.choice("model3",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model4":hp.choice("model4",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model5':hp.choice("model5",['KNN','LGBM','XGB','ET','GBDT','RF'])
                        }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(500)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn)
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
    model_list=['KNN','LGBM','XGB','ET','GBDT','RF']
    model1=model_list[params_best["model1"]]
    model2=model_list[params_best["model2"]]
    model3=model_list[params_best["model3"]]
    model4=model_list[params_best["model4"]]
    model5=model_list[params_best["model5"]]
    print("Metamodel is GBDT")
    print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
def ET_model(val):
    def hypeopt_objective(params):
        model1 = params["model1"]
        model2 = params["model2"]
        model3 = params["model3"]
        model4 = params["model4"]
        model5 = params["model5"]
    #model1
        if model1=='KNN':
            base_model1= KNN
        elif model1=='LGBM':
            base_model1=LGBM
        elif model1=='XGB':
            base_model1 =XGB
        elif model1=='ET':
            base_model1=ET
        elif model1=='GBDT':
            base_model1=GBDT
        elif model1=='RF':
            base_model1 =RF
    #model2
        if model2=='KNN':
            base_model2= KNN
        elif model2=='LGBM':
            base_model2=LGBM
        elif model2=='XGB':
            base_model2 = XGB
        elif model2=='ET':
            base_model2=ET
        elif model2=='GBDT':
            base_model2=GBDT
        elif model2=='RF':
            base_model2 = RF
    #model3
        if model3=='KNN':
            base_model3= KNN
        elif model3=='LGBM':
            base_model3=LGBM
        elif model3=='XGB':
            base_model3 = XGB
        elif model3=='ET':
            base_model3=ET
        elif model3=='GBDT':
            base_model3=GBDT
        elif model3=='RF':
            base_model3 =RF
    #model4
        if model4=='KNN':
            base_model4=KNN
        elif model4=='LGBM':
            base_model4=LGBM
        elif model4=='XGB':
            base_model4 =XGB
        elif model4=='ET':
            base_model4=ET
        elif model4=='GBDT':
            base_model4=GBDT
        elif model4=='RF':
            base_model4 =RF
    #model5
        if model5=='KNN':
            base_model5=KNN
        elif model5=='LGBM':
            base_model5=LGBM
        elif model5=='XGB':
            base_model5 =XGB
        elif model5=='ET':
            base_model5=ET
        elif model5=='GBDT':
            base_model5=GBDT
        elif model5=='RF':
            base_model5 =RF
        stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                          ,meta_regressor=ET
                                          ,shuffle=True
                                          ,n_jobs=-1
                                          ,random_state=2024)
        cv=KFold(n_splits=10,shuffle=True,random_state=2024)
        validation_loss=cross_validate(stack_gen
                                       ,x_train,y_train
                                       ,cv=cv
                                       ,verbose=False
                                       ,n_jobs=-1
                                       ,error_score='raise'
                                        )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'model1':hp.choice("model1",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model2':hp.choice("model2",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model3":hp.choice("model3",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model4":hp.choice("model4",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model5':hp.choice("model5",['KNN','LGBM','XGB','ET','GBDT','RF'])
                        }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(500)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn)
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
    model_list=['KNN','LGBM','XGB','ET','GBDT','RF']
    model1=model_list[params_best["model1"]]
    model2=model_list[params_best["model2"]]
    model3=model_list[params_best["model3"]]
    model4=model_list[params_best["model4"]]
    model5=model_list[params_best["model5"]]
    print("Metamodel is ET")
    print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
def XGB_model(val):
    def hypeopt_objective(params):
        model1 = params["model1"]
        model2 = params["model2"]
        model3 = params["model3"]
        model4 = params["model4"]
        model5 = params["model5"]
    #model1
        if model1=='KNN':
            base_model1= KNN
        elif model1=='LGBM':
            base_model1=LGBM
        elif model1=='XGB':
            base_model1 =XGB
        elif model1=='ET':
            base_model1=ET
        elif model1=='GBDT':
            base_model1=GBDT
        elif model1=='RF':
            base_model1 =RF
    #model2
        if model2=='KNN':
            base_model2= KNN
        elif model2=='LGBM':
            base_model2=LGBM
        elif model2=='XGB':
            base_model2 = XGB
        elif model2=='ET':
            base_model2=ET
        elif model2=='GBDT':
            base_model2=GBDT
        elif model2=='RF':
            base_model2 = RF
    #model3
        if model3=='KNN':
            base_model3= KNN
        elif model3=='LGBM':
            base_model3=LGBM
        elif model3=='XGB':
            base_model3 = XGB
        elif model3=='ET':
            base_model3=ET
        elif model3=='GBDT':
            base_model3=GBDT
        elif model3=='RF':
            base_model3 =RF
    #model4
        if model4=='KNN':
            base_model4=KNN
        elif model4=='LGBM':
            base_model4=LGBM
        elif model4=='XGB':
            base_model4 =XGB
        elif model4=='ET':
            base_model4=ET
        elif model4=='GBDT':
            base_model4=GBDT
        elif model4=='RF':
            base_model4 =RF
    #model5
        if model5=='KNN':
            base_model5=KNN
        elif model5=='LGBM':
            base_model5=LGBM
        elif model5=='XGB':
            base_model5 =XGB
        elif model5=='ET':
            base_model5=ET
        elif model5=='GBDT':
            base_model5=GBDT
        elif model5=='RF':
            base_model5 =RF
        stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                          ,meta_regressor=XGB
                                          ,shuffle=True
                                          ,n_jobs=-1
                                          ,random_state=2024)
        cv=KFold(n_splits=10,shuffle=True,random_state=2024)
        validation_loss=cross_validate(stack_gen
                                       ,x_train,y_train
                                       ,cv=cv
                                       ,verbose=False
                                       ,n_jobs=-1
                                       ,error_score='raise'
                                        )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'model1':hp.choice("model1",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model2':hp.choice("model2",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model3":hp.choice("model3",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model4":hp.choice("model4",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model5':hp.choice("model5",['KNN','LGBM','XGB','ET','GBDT','RF'])
                        }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(500)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn)
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
    model_list=['KNN','LGBM','XGB','ET','GBDT','RF']
    model1=model_list[params_best["model1"]]
    model2=model_list[params_best["model2"]]
    model3=model_list[params_best["model3"]]
    model4=model_list[params_best["model4"]]
    model5=model_list[params_best["model5"]]
    print("Metamodel is XGB")
    print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
def LGBM_model(val):
    def hypeopt_objective(params):
        model1 = params["model1"]
        model2 = params["model2"]
        model3 = params["model3"]
        model4 = params["model4"]
        model5 = params["model5"]
    #model1
        if model1=='KNN':
            base_model1= KNN
        elif model1=='LGBM':
            base_model1=LGBM
        elif model1=='XGB':
            base_model1 =XGB
        elif model1=='ET':
            base_model1=ET
        elif model1=='GBDT':
            base_model1=GBDT
        elif model1=='RF':
            base_model1 =RF
    #model2
        if model2=='KNN':
            base_model2= KNN
        elif model2=='LGBM':
            base_model2=LGBM
        elif model2=='XGB':
            base_model2 = XGB
        elif model2=='ET':
            base_model2=ET
        elif model2=='GBDT':
            base_model2=GBDT
        elif model2=='RF':
            base_model2 = RF
    #model3
        if model3=='KNN':
            base_model3= KNN
        elif model3=='LGBM':
            base_model3=LGBM
        elif model3=='XGB':
            base_model3 = XGB
        elif model3=='ET':
            base_model3=ET
        elif model3=='GBDT':
            base_model3=GBDT
        elif model3=='RF':
            base_model3 =RF
    #model4
        if model4=='KNN':
            base_model4=KNN
        elif model4=='LGBM':
            base_model4=LGBM
        elif model4=='XGB':
            base_model4 =XGB
        elif model4=='ET':
            base_model4=ET
        elif model4=='GBDT':
            base_model4=GBDT
        elif model4=='RF':
            base_model4 =RF
    #model5
        if model5=='KNN':
            base_model5=KNN
        elif model5=='LGBM':
            base_model5=LGBM
        elif model5=='XGB':
            base_model5 =XGB
        elif model5=='ET':
            base_model5=ET
        elif model5=='GBDT':
            base_model5=GBDT
        elif model5=='RF':
            base_model5 =RF
        stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                          ,meta_regressor=LGBM
                                          ,shuffle=True
                                          ,n_jobs=-1
                                          ,random_state=2024)
        cv=KFold(n_splits=10,shuffle=True,random_state=2024)
        validation_loss=cross_validate(stack_gen
                                       ,x_train,y_train
                                       ,cv=cv
                                       ,verbose=False
                                       ,n_jobs=-1
                                       ,error_score='raise'
                                        )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'model1':hp.choice("model1",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model2':hp.choice("model2",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model3":hp.choice("model3",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model4":hp.choice("model4",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model5':hp.choice("model5",['KNN','LGBM','XGB','ET','GBDT','RF'])
                        }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(500)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn)
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
    model_list=['KNN','LGBM','XGB','ET','GBDT','RF']
    model1=model_list[params_best["model1"]]
    model2=model_list[params_best["model2"]]
    model3=model_list[params_best["model3"]]
    model4=model_list[params_best["model4"]]
    model5=model_list[params_best["model5"]]
    print("Metamodel is LGBM")
    print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
def KNN_model(val):
    def hypeopt_objective(params):
        model1 = params["model1"]
        model2 = params["model2"]
        model3 = params["model3"]
        model4 = params["model4"]
        model5 = params["model5"]
    #model1
        if model1=='KNN':
            base_model1= KNN
        elif model1=='LGBM':
            base_model1=LGBM
        elif model1=='XGB':
            base_model1 =XGB
        elif model1=='ET':
            base_model1=ET
        elif model1=='GBDT':
            base_model1=GBDT
        elif model1=='RF':
            base_model1 =RF
    #model2
        if model2=='KNN':
            base_model2= KNN
        elif model2=='LGBM':
            base_model2=LGBM
        elif model2=='XGB':
            base_model2 = XGB
        elif model2=='ET':
            base_model2=ET
        elif model2=='GBDT':
            base_model2=GBDT
        elif model2=='RF':
            base_model2 = RF
    #model3
        if model3=='KNN':
            base_model3= KNN
        elif model3=='LGBM':
            base_model3=LGBM
        elif model3=='XGB':
            base_model3 = XGB
        elif model3=='ET':
            base_model3=ET
        elif model3=='GBDT':
            base_model3=GBDT
        elif model3=='RF':
            base_model3 =RF
    #model4
        if model4=='KNN':
            base_model4=KNN
        elif model4=='LGBM':
            base_model4=LGBM
        elif model4=='XGB':
            base_model4 =XGB
        elif model4=='ET':
            base_model4=ET
        elif model4=='GBDT':
            base_model4=GBDT
        elif model4=='RF':
            base_model4 =RF
    #model5
        if model5=='KNN':
            base_model5=KNN
        elif model5=='LGBM':
            base_model5=LGBM
        elif model5=='XGB':
            base_model5 =XGB
        elif model5=='ET':
            base_model5=ET
        elif model5=='GBDT':
            base_model5=GBDT
        elif model5=='RF':
            base_model5 =RF
        stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                          ,meta_regressor=KNN
                                          ,shuffle=True
                                          ,n_jobs=-1
                                          ,random_state=2024)
        cv=KFold(n_splits=10,shuffle=True,random_state=2024)
        validation_loss=cross_validate(stack_gen
                                       ,x_train,y_train
                                       ,cv=cv
                                       ,verbose=False
                                       ,n_jobs=-1
                                       ,error_score='raise'
                                        )
        return -np.mean(validation_loss["test_score"])
    param_grid_simple={
                        'model1':hp.choice("model1",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model2':hp.choice("model2",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model3":hp.choice("model3",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,"model4":hp.choice("model4",['KNN','LGBM','XGB','ET','GBDT','RF'])
                       ,'model5':hp.choice("model5",['KNN','LGBM','XGB','ET','GBDT','RF'])
                        }
    def param_hypeopt(max_evals):
        trials=Trials()
        early_stop_fn=no_progress_loss(500)
        params_best=fmin(hypeopt_objective
                        ,space=param_grid_simple
                        ,algo=tpe.suggest
                        ,max_evals=max_evals
                        ,verbose=True
                        ,trials=trials
                        ,early_stop_fn=early_stop_fn)
        print("\n","\n","best params:",params_best,"\n")
        return params_best,trials
    params_best,trials=param_hypeopt(val)
    model_list=['KNN','LGBM','XGB','ET','GBDT','RF']
    model1=model_list[params_best["model1"]]
    model2=model_list[params_best["model2"]]
    model3=model_list[params_best["model3"]]
    model4=model_list[params_best["model4"]]
    model5=model_list[params_best["model5"]]
    print("Metamodel is KNN")
    print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
val=500
RF_model(val)
GBDT_model(val)
ET_model(val)
XGB_model(val)
LGBM_model(val)
KNN_model(val)