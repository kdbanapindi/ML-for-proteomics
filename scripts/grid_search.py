#import analysis
import sys

import test_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import subplots,scatter
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import cross_val_score
#from yellowbrick.features import RFECV
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost

df_abd=pd.read_pickle('../Orbitrap_aplysia/abd.pkl')

psm_ID=list(pd.read_csv('../Aplysia_ganglia/Abdominal/DB search psm.csv')['Scan'])
y=np.zeros(df_abd.shape[0])

for i in range(0,df_abd.shape[0]):
    if df_abd.index[i] in psm_ID:
        y[i]=1
        
y =y.astype(int)
X = df_abd.drop('RT',axis=1).fillna(0).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

Cs=[10**-4,10**-3,10**-2,0.1,10,10**2,10**3,10**4]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma': gammas}

svm_rbf_best = test_features.SVM_classi(X_train, y_train, 'rbf', param_grid, True, 4)

