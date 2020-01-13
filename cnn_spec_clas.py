# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:13:03 2019

@author: Krishna
"""
import pandas as pd
import numpy as np
import seaborn as sns
import os

"""
Reading the data from the pickle file

"""
os.chdir('C:/Krishna/Box_sync/Box Sync/Research (anapind2@illinois.edu)/Projects/DL for MS/Orbitrap_aplysia')

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
 
y=y.astype(int)
"""
Performing image classification using pytorch
"""

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()

X_train, X_test,y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=2)

shape = (-1, 110, 110) #reshaping the input data into 110x110 image files
    

X_train = mm_scaler.fit_transform(np.float32(X_train)).reshape(shape)

X_test = mm_scaler.fit_transform(np.float32(X_test)).reshape(shape)


"""

Pytorch dataset 
"""

import torch
from torch.utils.data import DataLoader
    
X_train=torch.Tensor(X_train)
X_test=torch.Tensor(X_test)

y_train=torch.Tensor(y_train)
y_test=torch.Tensor(y_test)

"""
chacking for GPU
"""

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")



"""

Hyperparameters
"""

num_epochs = 5
num_classes = 2
batch_size = 100
learning_rate = 0.001


train_loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=False)
























