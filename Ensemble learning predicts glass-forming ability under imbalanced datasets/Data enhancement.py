import pandas as pd
import numpy as np
import smogn
import seaborn as sns
import resreg
df=pd.read_excel(r"C:\Users\cdj\Desktop\Dmax1531.xlsx")
x=df.iloc[:,:3]
y=df.iloc[:,-1:].values
y=y.ravel()
#wercs
relevance = resreg.sigmoid_relevance(y, cl=None, ch=3.5)
x_wercs,y_wercs=resreg.wercs(x,y,relevance,random_state=2024)
df_features = pd.DataFrame(x_wercs, columns=['Tg', 'Tx', 'Tl'])
df_labels = pd.DataFrame(y_wercs, columns=['Dmax'])
df_combined = pd.concat([df_features, df_labels], axis=1)
df_combined.to_excel('wercs.xlsx', index=False)
#smogn
df=pd.read_excel(r"C:\Users\cdj\Desktop\Dmax1531.xlsx")
x=df.iloc[:,:3]
y=df.iloc[:,-1:].values
y=y.ravel()
df2=smogn.smoter(data=df,y='Dmax')
df2.to_excel('smogn.xlsx', index=False)
#picture
# df_wercs=pd.read_excel(r"D:\桌面文件\研究生\研一上\学习笔记\机器学习\jupyter notebook save path\test3\wercs.xlsx")
# sns.distplot(df_wercs.Dmax)
# df_smogn=pd.read_excel(r"D:\桌面文件\研究生\研一上\学习笔记\机器学习\jupyter notebook save path\test3\smogn.xlsx")
# sns.distplot(df_smogn.Dmax)
# df=pd.read_excel(r"C:\Users\cdj\Desktop\Dmax1531.xlsx")
# sns.distplot(df.Dmax)
#PCD
def PCD(XR, XS):
    corr_XR = np.corrcoef(XR.T)
    corr_XS = np.corrcoef(XS.T)
    diff_corr = corr_XR - corr_XS
    PCD = np.linalg.norm(diff_corr, 'fro')
    return PCD
df=pd.read_excel(r"C:\Users\cdj\Desktop\Dmax1531.xlsx")
df_wercs=pd.read_excel(r"D:\桌面文件\研究生\研一上\学习笔记\机器学习\jupyter notebook save path\test3\wercs.xlsx")
df_smogn=pd.read_excel(r"D:\桌面文件\研究生\研一上\学习笔记\机器学习\jupyter notebook save path\test3\smogn.xlsx")
df=df.iloc[:,:]
df=df.values
df_wercs=df_wercs.iloc[:,:]
df_wercs=df_wercs.values
df_smogn=df_smogn.iloc[:,:]
df_smogn=df_smogn.values
pcd_wercs=PCD(df, df_wercs)
print("Wercs's PCD",pcd_wercs)
pcd_smogn=PCD(df, df_smogn)
print("Smogn's PCD",pcd_smogn)