import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import random


cd \Users\Mountassir Youssef\Desktop\challenge\data

train=pd.read_csv('train.csv')
train.columns=['Unnamed: 0', 'ID_1', 'ID_2', 'ID_T', 'ID_R', 'RESULT',
       'DATE']
train.head()
stat=pd.read_csv('stats.csv')
stat.head()
stat.columns
stat_1=stat[['ID1','RPW_1', 'RPWOF_1']]
stat_1.columns=['ID_1', 'RPW_1', 'RPWOF_1']
stat_1.head()
stat_2=stat[['ID2','RPW_2', 'RPWOF_2']]
stat_2.columns=['ID_2', 'RPW_2', 'RPWOF_2']
stat_2.head()
test=pd.read_csv('test.csv')
test.columns=['ID_1', 'ID_2', 'ID_T', 'ID_R', 'DATE']
test.head()
tour=pd.read_csv('tour.csv')
tour_mod=tour[['ID_T','DATE_T','RANK_T']]
tour_mod.columns=['ID_T', 'DATE', 'RANK_T']
tour_mod.head()
player=pd.read_csv('players.csv')
player.head()
player_mod=player[['ID_P','NAME_P']]
player_mod.head()
merge_1=pd.merge(left=train.head(2000),right=stat_1.head(2000),left_on='ID_1',right_on='ID_1',how='inner')
merge_1.head()
merge_t_1=pd.merge(left=test.head(2000),right=stat_1.head(2000),left_on='ID_1',right_on='ID_1',how='inner')
merge_t_1.head()
merge_t_2=pd.merge(left=merge_t_1.head(2000),right=stat_2.head(2000),left_on='ID_2',right_on='ID_2',how='inner')
merge_t_2.head()
merge = merge_t_2[['ID_1','ID_2','ID_T','ID_R','RPW_1','RPWOF_1','RPW_2','RPWOF_2']]
merge.head()
merge_t=merge_t_2[['ID_1','ID_2','ID_T','ID_R','RPW_1','RPWOF_1','RPW_2','RPWOF_2']]
merge_t.head()

merge.head()
merge.columns
X=merge[['ID_1', 'ID_2', 'ID_T', 'ID_R', 'RPW_1', 'RPWOF_1', 'RPW_2', 'RPWOF_2']].values
X_t=merge_t[['ID_1', 'ID_2', 'ID_T', 'ID_R', 'RPW_1', 'RPWOF_1', 'RPW_2', 'RPWOF_2']].values

y=merge['RESULT'].values
y

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# X_t is the array from test data frame on which we want to test our model
X_t = preprocessing.StandardScaler().fit(X_t).transform(X_t.astype(float))

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split( X[0:1000], y[0:1000], test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

k = 1
while k<11:
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = 7).fit(X_train,y_train)
    neigh
    yhat = neigh.predict(X_test)
    yhat
    from sklearn import metrics
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
    k=k+1


#k=7 is the best

y_test_hat = neigh.predict(X_t)
y_test_hat

test['R']= pd.Series(y_test_hat[0:6097], index=test.index)
test


test.to_csv(r'\Users\Mountassir Youssef\Desktop\challenge\data\ results.csv')

