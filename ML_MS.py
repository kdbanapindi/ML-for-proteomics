# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:28:54 2019

@author: Krishna
"""

"""
Reading the data from the pickle file

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

os.chdir('C:/Krishna/Box_sync/Box Sync/Research (anapind2@illinois.edu)/Projects/DL for MS/Orbitrap_aplysia')


df_abd=pd.read_pickle('./abd.pkl')


#parsing the scan values

os.chdir('C:/Krishna/Box_sync/Box Sync/Research (anapind2@illinois.edu)/Projects/DL for MS/Aplysia_ganglia/Abdominal')

psm_ID=list(pd.read_csv('DB search psm.csv')['Scan'])

y=np.zeros(df_abd.shape[0])

for i in range(0,df_abd.shape[0]):
    
    if df_abd.index[i] in psm_ID:
        
        y[i]=1
 

df_abd['y']=y
   
df_cer=df_cer.set_index(df_cer.index.astype(str) + '_cer')

df_all=(pd.concat([df_abd,df_buc,df_cer], axis=0, sort=False)).fillna(0)


df_all.to_pickle('./abd_buc_cer.pkl')
#classifying based on SVM
        
from sklearn import svm    
from sklearn.model_selection import train_test_split    
from sklearn.metrics import classification_report 

df_abd=pd.read_pickle('./abd.pkl')

X=df_abd.iloc[:,0:-1].fillna(0)
y=df_abd.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf_svm = svm.SVC(gamma='scale', kernel='rbf', class_weight='balanced',
                  random_state=0)

mod_svm=clf_svm.fit(X_train,y_train)

y_pred_svm=mod_svm.predict(X_test)



print(classification_report(y_test, y_pred_svm))


#random forest

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

mod_rf=clf_rf.fit(X_train, y_train) 

y_pred_rf=mod_rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))
"""
Logistic regression

"""


import numpy as np
red_len=np.random.randint(0,df_all.shape[0],1000)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split   
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 


scaler=MinMaxScaler()

df_all_sub=df_all.iloc[list(red_len),:]

sc_par=scaler.fit(df_all)

df_all_sub=pd.DataFrame(scaler.transform(df_all_sub))



X=df_all.iloc[:,0:-1]
y=df_all.iloc[:,-1]

#reduced 
clf_lr=LogisticRegression(penalty='elasticnet',random_state=0,class_weight={0.:1,1.:2},
                          solver='saga', multi_class='auto', l1_ratio=0.5)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)




mod_lr=clf_lr.fit(X_train, y_train)


y_pred_lr=mod_lr.predict(X_test)

print(classification_report(y_test, y_pred_lr))

"""
SVM
"""

clf_svm = svm.SVC(gamma='scale', kernel='rbf', class_weight='balanced',
                  random_state=0)

mod_svm=clf_svm.fit(X_train,y_train)
y_pred_svm=mod_svm.predict(X_test)



print(classification_report(y_test, y_pred_svm))


#ensemble model

y_ens_pred=np.zeros(len(y_test))

for i in range(0, len(y_test)):
    
    if y_pred_svm[i]+ y_pred_lr[i]+y_pred_rf[i]>0:
        y_ens_pred[i]=1
    
print(classification_report(y_test, y_ens_pred))    


#LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf_lda=LinearDiscriminantAnalysis()

mod_lda=clf_lda.fit(X_train,y_train)

y_pred_lds=mod_lda.fit(X_train,y_train)
