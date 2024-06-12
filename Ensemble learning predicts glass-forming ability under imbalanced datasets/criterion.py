import numpy as np
import pandas as pd
df=pd.read_excel(r"C:\Users\cdj\Desktop\Dmax1531.xlsx")
tg=df['Tg']
tx=df['Tx']
tl=df['Tl']
y=df['Dmax']
def correlation_coefficient(y_true, y_pred):
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return correlation
formula1 = tg/tl#Trg
formula2 = tx/(tg+tl)#γ
formula3 = tx/tg+tg/tl#β1
formula4 = tg/tx-tg/(1.3*tl)#β
formula5 = (tg/tl)*pow((tx-tg)/tg,0.143)#Φ
formula6 =tx*tg/pow(tl-tx,2)#β2
formula7 =(tx-tg)/(tl-tg)#ΔTrg
formula8 =(3*tx-2*tg)/tl#γc
formula9 =tl*(tl+tx)/(tx*(tl-tx))#ω
formula10 =(tx-tg)/(tl-tx)*pow(tx/(tl-tx),1.47)#χ
formula11 =tg*(tx-tg)/pow(tl-tx,2)#Gp
formula12=tx/(tl-tg)#δ
formula13=tx-tg#ΔTx
formula14=(2*tx-tg)/tl#γm
formula15=(2*tx-tg)/(tl+tx)#ωB
formula16=tx*tg*(tx-tg)/pow(tl-tx,3)#ν
formula17=((tg*pow(tx-tg,2))/(pow(tl-tx,3)))*pow(((tx-tg)/(tl-tg))-(tx/(tl+tx)),2)#θ
formulas = [formula1, formula2, formula3, formula4, formula5, formula6, formula7, formula8, formula9, formula10, formula11,formula12,formula13,formula14,formula15,formula16,formula17]
pearson=[]
#delete nan values and inf values
for formula in formulas:
    data = np.c_[formula,y]
    data = data[~np.isnan(data).any(axis=1)]
    data = data[~np.isinf(data).any(axis=1)]
    first_column = [row[0] for row in data]
    first_column_y = [row[1] for row in data]
    formula_pearson =correlation_coefficient(first_column,first_column_y)
    pearson.append(formula_pearson)
print(pearson)