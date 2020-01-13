# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:46:39 2019

@author: krish
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
os.chdir('C:/Krishna/Box_sync/Box Sync/Research (anapind2@illinois.edu)/Projects/DL for MS/Orbitrap_aplysia')

import pymzml

msrun=pymzml.run.Reader("Abdominal.mzML", MS_precisions={1:1e-5, 2:1e-4})

col_names=list(np.round(np.linspace(300,1500,12001),1))
col_names.append('RT')
df_all_spectra=pd.DataFrame(columns=col_names)

spec_ID=[]
spec_RT=[]
all_data=[]

count=0
for spectrum in msrun:  
    
    count+=1
    if count%1000==999:
        print(count+1)
    
    if spectrum.ms_level == 2:
        
        #spec_ID.append(int(spectrum.ID))
        #spec_RT.append(spectrum.scan_time[0])
        
        if spectrum.peaks('centroided').any():
            
            df_spect = pd.DataFrame(spectrum.peaks('centroided'),columns=['m/z',spectrum.ID])
            df_spect['m/z'] = round(df_spect['m/z'],0)
        
            indexNames = df_spect[(df_spect['m/z']<300) | (df_spect['m/z']>1500) ].index
            df_spect.drop(indexNames , inplace=True)
        
            df_spect=df_spect.groupby('m/z')[spectrum.ID].apply(lambda x: sum(x)).reset_index()
            df_spect.set_index('m/z', inplace=True)
            df_spect=df_spect.T
            df_spect['RT']=spectrum.scan_time[0]
            df_spect=df_spect.squeeze()
            
            all_data.append(df_spect)
        
            #df_all_spectra=df_all_spectra.append(df_spect)
        
        
     
            #print(spectrum.ID)
               


all_data=list(filter(lambda x : type(x) == pd.Series, all_data))

all_data_df=pd.DataFrame(all_data)

df_all_spectra=df_all_spectra.append(all_data_df)


#writing the data into a pkl file

df_all_spectra.to_pickle('./abd_2.pkl')
#parsing the PSM data



"""
Reading the data from the pickle file

"""

df_abd=pd.read_pickle('./abd.pkl')


#padding with extra zeros to make a dimension of (110x110)

dummy_cols=list(np.round(np.linspace(1502,1599,98),0))

df_dummy=pd.DataFrame(index=df_abd.index, columns=dummy_cols).fillna(0)


#concatenating the two dfs

df_abd=(pd.concat([df_abd,df_dummy], axis=1, sort=False)).fillna(0)



X=df_abd



#parsing the scan values

os.chdir('C:/Krishna/Box_sync/Box Sync/Research (anapind2@illinois.edu)/Projects/DL for MS/Aplysia_ganglia/Abdominal')

psm_ID=list(pd.read_csv('DB search psm.csv')['Scan'])

y=np.zeros(df_abd.shape[0])

for i in range(0,df_abd.shape[0]):
    
    if df_abd.index[i] in psm_ID:
        
        y[i]=1
 

"""
Performing image classification using pytorch
"""











       
#classifying based on SVM
        
from sklearn import svm    
from sklearn.model_selection import train_test_split    

X=df_all_spectra.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf_svm = svm.SVC(gamma='scale')

mod_svm=clf_svm.fit(X_train,y_train)
y_pred_svm=mod_svm.predict(X_test)

#evaluating the classification accuracy
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


#random forest

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

mod_rf=clf_rf.fit(X_train, y_train) 

y_pred_rf=mod_rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))
#logistic regression

from sklearn.linear_model import LogisticRegression

clf_lr=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

mod_lr=clf_lr.fit(X_train, y_train)


y_pred_lr=mod_lr.predict(X_test)

print(classification_report(y_test, y_pred_rf))

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
