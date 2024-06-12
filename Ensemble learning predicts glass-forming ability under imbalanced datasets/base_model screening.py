import pandas as pd
import numpy as np
import math
from minepy import MINE
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score,cross_validate
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
df=pd.read_excel(r"C:\Users\cdj\Desktop\smogn.xlsx")
x=df.iloc[:,:3]
y=df.iloc[:,-1:].values.ravel()
list_size=len(x)
test_size=0.3
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=2024)
DT=DecisionTreeRegressor(random_state=2024).fit(x_train,y_train)
svr=SVR().fit(x_train,y_train)
LR=LinearRegression().fit(x_train,y_train)
KNN= KNeighborsRegressor().fit(x_train,y_train)
LGBM=lgb.LGBMRegressor().fit(x_train,y_train)
XGB=XGBRegressor().fit(x_train,y_train)
RF=RandomForestRegressor().fit(x_train,y_train)
ET=ExtraTreesRegressor().fit(x_train,y_train)
GBDT=GradientBoostingRegressor().fit(x_train,y_train)
# model1_predictions = corss_train_test_predict(LR,x_train,y_train)
# model2_predictions = corss_train_test_predict(DT,x_train,y_train)
# model3_predictions = corss_train_test_predict(svr,x_train,y_train)
# model4_predictions = corss_train_test_predict(KNN,x_train,y_train)
# model5_predictions = corss_train_test_predict(LGBM,x_train,y_train)
# model6_predictions = corss_train_test_predict(XGB,x_train,y_train)
# model7_predictions = corss_train_test_predict(ET,x_train,y_train)
# model8_predictions = corss_train_test_predict(GBDT,x_train,y_train)
# model9_predictions = corss_train_test_predict(RF,x_train,y_train)
model1_predictions = LR.predict(x_test)
model2_predictions = DT.predict(x_test)
#model3_predictions = svr.predict(x_test)
model4_predictions = KNN.predict(x_test)
model5_predictions = LGBM.predict(x_test)
model6_predictions = XGB.predict(x_test)
model7_predictions = ET.predict(x_test)
model8_predictions = GBDT.predict(x_test)
model9_predictions = RF.predict(x_test)
def corss_train_test_predict(model,x_train,y_train):
    kf = KFold(n_splits=10, shuffle=True, random_state=2024)
    sum_array_size=list_size*(1-test_size)/kf.n_splits+1
    sum_array_size = math.floor(sum_array_size)
    sum_array = np.zeros(sum_array_size)
    for train_index, test_index in kf.split(x_train):
        x_train_oof, x_test_oof = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_oof, y_test_oof = y_train[train_index], y_train[test_index]
        y_train_oof = np.ravel(y_train_oof)
        y_test_oof = np.ravel(y_test_oof)
        model.fit(x_train_oof, y_train_oof)
        y_pred_oof = model.predict(x_test_oof)
        sum_array[:len(y_pred_oof)] += y_pred_oof
    mean_array = sum_array / kf.n_splits
    mean_array[164]=mean_array[164]*10/2
    return mean_array
predictions = [model1_predictions, model2_predictions,
               #model3_predictions
               model4_predictions,
               model5_predictions, model6_predictions,
               model7_predictions, model8_predictions,
               model9_predictions]
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15
mic_values = []
for i in range(len(predictions)):
    tmp = []
    for j in range(len(predictions)):
        m = MINE()
        m.compute_score(predictions[i], predictions[j])
        tmp.append(m.mic())
    mic_values.append(tmp)
mic_matrix = np.array(mic_values)
fig, ax = plt.subplots(figsize=(8,8))
def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap,vmin=0,vmax=1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(8)
    plt.xticks(tick_marks, ['LR','DT','KNN', 'LGBM', 'XGBoost','ET', 'GBDT','RF'], rotation=45,fontweight='bold')
    plt.yticks(tick_marks, ['LR','DT','KNN', 'LGBM', 'XGBoost','ET', 'GBDT','RF'],fontweight='bold')
    plt.tight_layout()
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
plot_confusion_matrix(mic_matrix
                      , title=''#'MIC values between regression models'
                     )
plt.savefig('mic_matrix.png', dpi=1200)
plt.show()
def sum_array(arr):
    result = 0
    for num in arr:
        result += num
    return result/8
mic_values=[]
for i in range(8):
    array_sum = sum_array(mic_matrix[i])
    mic_values.append(array_sum)
model_list = [LR,DT,KNN,LGBM,XGB,ET,GBDT,RF]
R2_train_test_list=[]
# for model in model_list:
#     R2_train_test=cross_val_score(model,x_train,y_train,cv=10,n_jobs=-1)
#     R2_train_test_score=R2_train_test.mean()
#     R2_train_test_list.append(R2_train_test_score)
for model in model_list:
    R2_train_test=model.score(x_test,y_test)
    R2_train_test_list.append(R2_train_test)
mic_values= np.array(mic_values)
R2_train_test_list= np.array(R2_train_test_list)
result=R2_train_test_list*0.6+(1-mic_values)*0.4
result.tolist()
print(R2_train_test_list,mic_values)